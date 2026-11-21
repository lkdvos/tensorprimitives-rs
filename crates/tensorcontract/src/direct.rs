//! Unpacked execution for contractions too small or too narrow for packing to
//! pay off.
//!
//! The operand with the smaller free extent is gathered onto the stack; the
//! other is streamed in chunks of [`W`] along its free extent, and each output
//! chunk is accumulated in registers. No heap allocation, tile buffer or
//! separate write-back.
//!
//! Results differ from the packed path in rounding ([`Real::mul_add`] fuses
//! only where FMA is baseline; complex products are summed as four real
//! planes). The path choice does not depend on the thread count, so results
//! stay identical across thread counts.

use core::mem::MaybeUninit;

use crate::element::{Element, Real};
use crate::plan::Plan;

/// Capacity, in reals per plane, of the gathered operand (`k` x its extent).
const GATHER_CAP: usize = 1024;

/// Accumulator lanes: two NEON/AVX2 vectors or one AVX-512 vector.
const W: usize = 8;

/// Largest `k` the path accepts.
const K_LINE: usize = 128;

/// Selection thresholds, fitted on an Apple M5 (see `docs/notebook/
/// small-contractions.md`): direct when `macs <= MAX_MACS`, or when
/// `min(m, n) <= MAX_MIN_DIM` and `k <= MAX_K`.
const MAX_MACS: u64 = 4096;
const MAX_MIN_DIM: usize = 4;
const MAX_K: usize = 32;

#[inline(always)]
fn unit_run(offs: &[i64]) -> bool {
    let o0 = offs[0];
    let mut ok = true;
    for (t, &o) in offs.iter().enumerate() {
        ok &= o == o0 + t as i64;
    }
    ok
}

/// Initialises only the prefix a call uses; zeroing the whole buffer is a
/// visible cost at these sizes.
fn init_prefix<R: Copy>(buf: &mut [MaybeUninit<R>], len: usize, z: R) -> &mut [R] {
    let pre = &mut buf[..len];
    for s in pre.iter_mut() {
        s.write(z);
    }
    // SAFETY: every element of `pre` was just initialised, and `MaybeUninit<R>`
    // has the same layout as `R`.
    unsafe { &mut *(pre as *mut [MaybeUninit<R>] as *mut [R]) }
}

/// An explicit [`Plan::with_blocking`] keeps the packed path: blocking only
/// means something to its loop nest.
pub(crate) fn use_direct(plan: &Plan) -> bool {
    let s = &plan.stats;
    if plan.blocking.is_some() || s.k == 0 || s.m == 0 || s.n == 0 || s.batch == 0 {
        return false;
    }
    let min_dim = s.m.min(s.n);
    s.k <= K_LINE
        && min_dim * s.k <= GATHER_CAP
        && (s.macs() <= MAX_MACS || (min_dim <= MAX_MIN_DIM && s.k <= MAX_K))
}

/// Vectorises along the larger free extent so lanes stay full when the other
/// is tiny (an MPS gate has `m = 4`).
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
    let gather_a = m <= n;
    // `g*`: gathered operand (extent `gn`); `s*`: streamed operand (extent `sn`).
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
    let (d_x, d_y, c_x, c_y) = if gather_a {
        (&plan.d_m, &plan.d_n, &plan.c_m, &plan.c_n)
    } else {
        (&plan.d_n, &plan.d_m, &plan.c_n, &plan.c_m)
    };
    let (gn, sn) = if gather_a { (m, n) } else { (n, m) };

    let z = T::Real::ZERO;
    let cplx = T::IS_COMPLEX;
    let read_c = beta != T::zero();
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
        // Gathered operand: row `x` at `[x*k..]`.
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
            // Streamed chunk: lane `t` of step `p` at `[p*W + t]`. Lanes past
            // `w` hold stale values and are never stored. Consecutive offsets
            // are loaded as one run so the loads vectorise.
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
                // Complex: four real accumulators, so every step is one FMA.
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
