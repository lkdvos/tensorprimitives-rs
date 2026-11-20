//! Direct (unpacked) execution for contractions too shallow to pay for packing.
//!
//! The five-loop driver earns its keep by amortising one packing pass over many
//! micro-kernel calls. That amortisation is per unit of contraction depth: a
//! panel of `A` packed once is reused across every column of `B`, but each
//! packed element feeds only `k` multiply-adds per output element. At small `k`
//! (an MPS two-site gate on qubits has `k = 4`) packing, the tile buffer and the
//! separate write-back pass dominate, and a plain loop over the scatter vectors
//! beats the whole engine — measured at about 14x on a 64-byte problem and
//! still about 1.4x at `n = 4096`.
//!
//! The loop here gathers the operand with the smaller free extent whole into
//! stack buffers, streams the other in chunks of [`W`], and accumulates each
//! output chunk in registers (see [`execute`]). No heap allocation, no kernel
//! dispatch, no tile buffer and no separate write-back pass. It is not a
//! micro-kernel: it is for the regime where the micro-kernel is not the
//! bottleneck, and the `auto` rule ([`Rule`]) keeps it to that regime.
//!
//! Selection is by [`use_direct`], gated by `TENSORCONTRACT_DIRECT`
//! (`auto` | `off` | `on`) so both arms can be A/B'd in one process restart.
//! `on` forces the path for every non-empty contraction whose gather block fits,
//! which is how the whole test suite is run through it.
//!
//! **Not bitwise identical to the packed path.** Each output element is still
//! accumulated by one thread over the full contracted extent in the plan's `K`
//! order, but through [`Real::mul_add`], which fuses only where FMA is baseline
//! for the build target (aarch64, or x86 built with `+fma`), and complex
//! products are summed as four real planes. Same class of difference as
//! `TENSORCONTRACT_KERNEL=scalar`. The choice of path is independent of the
//! thread count, so the `THREADS` bitwise-identity promise still holds.

use core::mem::MaybeUninit;

use crate::element::{Element, Real};
use crate::plan::Plan;

/// Elements per plane in the stack gather buffer: the gathered operand block
/// (`k` x the vectorised extent) must fit in it. 1024 `f64` per plane, two
/// planes plus two accumulator planes, is 32 KiB of stack.
const GATHER_CAP: usize = 1024;

/// Accumulator lanes per register block. Eight reals is two NEON or AVX2
/// vectors, one AVX-512 vector; enough to hide FMA latency at `k >= 2`
/// without spilling, for both planes of a complex accumulator.
const W: usize = 8;

/// Longest streamed line, i.e. the largest `k` this path accepts at all.
const K_LINE: usize = 128;

/// Whether `offs` is a run of consecutive offsets, `offs[t] = offs[0] + t`.
#[inline(always)]
fn unit_run(offs: &[i64]) -> bool {
    let o0 = offs[0];
    let mut ok = true;
    for (t, &o) in offs.iter().enumerate() {
        ok &= o == o0 + t as i64;
    }
    ok
}

/// Write `z` to the first `len` slots and hand them back as initialised.
fn init_prefix<R: Copy>(buf: &mut [MaybeUninit<R>], len: usize, z: R) -> &mut [R] {
    let pre = &mut buf[..len];
    for s in pre.iter_mut() {
        s.write(z);
    }
    // SAFETY: every element of `pre` was just initialised, and `MaybeUninit<R>`
    // has the same layout as `R`.
    unsafe { &mut *(pre as *mut [MaybeUninit<R>] as *mut [R]) }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Auto,
    Off,
    On,
}

fn mode() -> Mode {
    env_once!(Mode, "TENSORCONTRACT_DIRECT", Mode::Auto, |v: &str| match v
        .to_ascii_lowercase()
        .as_str()
    {
        "off" | "0" | "false" => Mode::Off,
        "on" | "1" | "true" | "force" => Mode::On,
        _ => Mode::Auto,
    })
}

/// The `auto` rule: direct when the whole contraction is at most
/// `TENSORCONTRACT_DIRECT_MACS` multiply-adds (default 4096), or when the
/// smaller free extent is at most `TENSORCONTRACT_DIRECT_MINDIM` (default 4)
/// and `k` is at most `TENSORCONTRACT_DIRECT_K` (default 32).
///
/// Fitted on an Apple M5 (one machine, so a per-microarchitecture claim, A56)
/// against a 648-cell grid of column-major GEMMs, `m` 1..64, `n` 8..4096,
/// `k` 1..32, in `c64` and `f64`: within 2.1% (`c64`) and 1.2% (`f64`) geomean
/// of choosing the faster arm per cell, where the packed path alone is 38%
/// and 39% off it (re-fitted after the unit-stride chunk paths). The grid alone prefers a minimum dimension of 3, because
/// `m = 4` is exactly this machine's `c64` register block; 4 is chosen because
/// on the MPS two-site gate (`m = 4`, `k = 4`, output strided along `N`) the
/// direct path is ~10% faster than the packed one at `n` = 361 and 4096, and
/// the grid gives up only ~1% for it.
#[derive(Clone, Copy)]
struct Rule {
    macs: u64,
    min_dim: usize,
    k: usize,
}

fn rule() -> Rule {
    Rule {
        macs: env_once!(u64, "TENSORCONTRACT_DIRECT_MACS", 4096, |v: &str| v
            .parse()
            .unwrap_or(4096)),
        min_dim: env_once!(usize, "TENSORCONTRACT_DIRECT_MINDIM", 4, |v: &str| v
            .parse()
            .unwrap_or(4)),
        k: env_once!(usize, "TENSORCONTRACT_DIRECT_K", 32, |v: &str| v
            .parse()
            .unwrap_or(32)),
    }
}

/// Whether [`execute`] should take this plan. Pure function of the plan and the
/// switches, so it is decided identically on every call.
pub(crate) fn use_direct(plan: &Plan) -> bool {
    let s = &plan.stats;
    if s.k == 0 || s.m == 0 || s.n == 0 || s.batch == 0 {
        return false;
    }
    let fits = s.k <= K_LINE && s.m.min(s.n) * s.k <= GATHER_CAP;
    match mode() {
        Mode::Off => false,
        Mode::On => fits,
        // Deliberately independent of the thread count: `THREADS` is promised
        // to be bitwise identical at any value, and this path rounds differently
        // from the packed one, so the choice between them must not move with it.
        Mode::Auto => {
            let r = rule();
            fits && (s.macs() <= r.macs || (s.m.min(s.n) <= r.min_dim && s.k <= r.k))
        }
    }
}

/// Run the contraction with no packing.
///
/// The operand with the smaller free extent is gathered whole, once per batch,
/// into planar stack buffers; the other is streamed in chunks of [`W`] along its
/// free extent, each chunk gathered `k`-major so a chunk column is one vector.
/// Every output is then `k` broadcast-multiply-adds of a gathered scalar into a
/// `W`-wide register accumulator. Vectorising along the *larger* free extent
/// keeps the lanes full: an MPS gate has `m = 4`, and padding that to a vector
/// wasted half of every one.
///
/// # Safety
///
/// As [`crate::driver::execute`]; additionally [`use_direct`] must have
/// returned true for `plan` (so `k >= 1` and the gathered block fits).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn execute<T: Element>(
    plan: &Plan,
    alpha: T,
    a: *const T,
    b: *const T,
    beta: T,
    c: *const T,
    d: *mut T,
) {
    let (m, n, k) = (plan.stats.m, plan.stats.n, plan.stats.k);
    // Gather the `M` side (`A`) whole when it is the smaller, and stream `B`.
    let gather_a = m <= n;
    // `g*`: gathered operand, extent `gn`. `s*`: streamed operand, extent `sn`.
    let (g_ptr, g_x, g_k, g_conj, gh) = if gather_a {
        (a, &plan.a_m, &plan.a_k, plan.conj_a, &plan.h_a)
    } else {
        (b, &plan.b_n, &plan.b_k, plan.conj_b, &plan.h_b)
    };
    let (s_ptr, s_y, s_k, s_conj, sh) = if gather_a {
        (b, &plan.b_n, &plan.b_k, plan.conj_b, &plan.h_b)
    } else {
        (a, &plan.a_m, &plan.a_k, plan.conj_a, &plan.h_a)
    };
    // Output offsets along the gathered (`x`) and streamed (`y`) extents.
    let (d_x, d_y, c_x, c_y) = if gather_a {
        (&plan.d_m, &plan.d_n, &plan.c_m, &plan.c_n)
    } else {
        (&plan.d_n, &plan.d_m, &plan.c_n, &plan.c_m)
    };
    let (gn, sn) = if gather_a { (m, n) } else { (n, m) };

    let z = T::Real::ZERO;
    let cplx = T::IS_COMPLEX;
    let read_c = beta != T::zero();
    // Uninitialised, then only the prefix this call uses is written: zeroing
    // the full buffers was a visible cost at the sizes this path exists for.
    let mut gr_s = [MaybeUninit::<T::Real>::uninit(); GATHER_CAP];
    let mut gi_s = [MaybeUninit::<T::Real>::uninit(); GATHER_CAP];
    let mut cr_s = [MaybeUninit::<T::Real>::uninit(); K_LINE * W];
    let mut ci_s = [MaybeUninit::<T::Real>::uninit(); K_LINE * W];
    let gr = init_prefix(&mut gr_s, gn * k, z);
    let gi = init_prefix(&mut gi_s, if cplx { gn * k } else { 0 }, z);
    let cr = init_prefix(&mut cr_s, k * W, z);
    let ci = init_prefix(&mut ci_s, if cplx { k * W } else { 0 }, z);

    let parts = |x: T, conj: bool| {
        let im = x.im();
        (x.re(), if conj { -im } else { im })
    };

    for h in 0..plan.stats.batch {
        let (g0, s0) = (gh[h] as isize, sh[h] as isize);
        let ch = plan.h_c.get(h).copied().unwrap_or(0) as isize;
        let dh = plan.h_d[h] as isize;
        // Gathered operand, row `x` at `[x*k..]`.
        for x in 0..gn {
            let gx = g_x[x];
            for p in 0..k {
                let (re, im) = parts(*g_ptr.offset(g0 + (gx + g_k[p]) as isize), g_conj);
                gr[x * k + p] = re;
                if cplx {
                    gi[x * k + p] = im;
                }
            }
        }
        for y0 in (0..sn).step_by(W) {
            let w = W.min(sn - y0);
            // Streamed chunk, `k`-major: lane `t` of step `p` at `[p*W + t]`.
            // Lanes past `w` keep whatever they held; they are never stored.
            // A full chunk whose streamed offsets are consecutive is a
            // contiguous run per `k` step: load it as one, so the loads and
            // the de-interleave vectorise instead of going through the
            // scatter table per element.
            let s_unit = w == W && unit_run(&s_y[y0..y0 + W]);
            if s_unit {
                let sy = s_y[y0];
                for p in 0..k {
                    let base = s_ptr.offset(s0 + (sy + s_k[p]) as isize);
                    for t in 0..W {
                        let (re, im) = parts(*base.add(t), s_conj);
                        cr[p * W + t] = re;
                        if cplx {
                            ci[p * W + t] = im;
                        }
                    }
                }
            } else {
                for t in 0..w {
                    let sy = s_y[y0 + t];
                    for p in 0..k {
                        let (re, im) = parts(*s_ptr.offset(s0 + (sy + s_k[p]) as isize), s_conj);
                        cr[p * W + t] = re;
                        if cplx {
                            ci[p * W + t] = im;
                        }
                    }
                }
            }
            let d_unit = w == W && !read_c && !plan.conj_d && unit_run(&d_y[y0..y0 + W]);
            for x in 0..gn {
                // Complex: four real accumulators (the 4m form), so every
                // step is a fused multiply-add and the two subtractions happen
                // once per output rather than once per `k`.
                let mut ar = [z; W];
                let mut ai = [z; W];
                let mut br = [z; W];
                let mut bi = [z; W];
                let grx = &gr[x * k..x * k + k];
                if cplx {
                    let gix = &gi[x * k..x * k + k];
                    for p in 0..k {
                        let (vr, vi) = (grx[p], gix[p]);
                        let (rr, ri) = (&cr[p * W..p * W + W], &ci[p * W..p * W + W]);
                        for t in 0..W {
                            ar[t] = vr.mul_add(rr[t], ar[t]);
                            br[t] = vi.mul_add(ri[t], br[t]);
                            ai[t] = vr.mul_add(ri[t], ai[t]);
                            bi[t] = vi.mul_add(rr[t], bi[t]);
                        }
                    }
                    for t in 0..W {
                        ar[t] = ar[t] - br[t];
                        ai[t] += bi[t];
                    }
                } else {
                    for p in 0..k {
                        let vr = grx[p];
                        let rr = &cr[p * W..p * W + W];
                        for t in 0..W {
                            ar[t] = vr.mul_add(rr[t], ar[t]);
                        }
                    }
                }
                let (dx, cx) = (d_x[x], c_x.get(x).copied().unwrap_or(0));
                if d_unit && alpha == T::one() {
                    // Contiguous output run, plain store: one interleaving
                    // vector store per lane pair instead of a scatter.
                    let base = d.offset(dh + (dx + d_y[y0]) as isize);
                    for t in 0..W {
                        *base.add(t) = T::from_parts(ar[t], ai[t]);
                    }
                    continue;
                }
                for t in 0..w {
                    let y = y0 + t;
                    let mut val = alpha.mul(T::from_parts(ar[t], ai[t]));
                    if read_c {
                        let coff = ch + (cx + c_y.get(y).copied().unwrap_or(0)) as isize;
                        let cv = *c.offset(coff);
                        let cv = if plan.conj_c { cv.conj() } else { cv };
                        val = val.add(beta.mul(cv));
                    }
                    if plan.conj_d {
                        val = val.conj();
                    }
                    *d.offset(dh + (dx + d_y[y]) as isize) = val;
                }
            }
        }
    }
}
