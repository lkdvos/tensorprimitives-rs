//! Dot products along `K` for contractions with a small free extent on one
//! side and a deep `K`.
//!
//! The register tile vectorises along `M`, so with `m` (or `n`) far below `MR`
//! most lanes are padding, and packing a `K`-major operand into `M`-major
//! panels costs as much as the arithmetic. Here one operand ("streamed") must
//! have a unit-stride `K` axis and is read in place; the other ("gathered") is
//! copied `K`-major, one `K` block at a time. Every output is a dot product
//! vectorised along `K`, reduced horizontally once per block.
//!
//! Results differ from the packed path in rounding (summation order along
//! `K`). The path ignores the thread count, so results stay identical across
//! thread counts, and it runs serially even when threads were asked for.

use core::mem::MaybeUninit;

use crate::element::{Element, Real};
use crate::kernel::{KernelSet, DOT_NB as NB};
use crate::plan::Plan;

/// Reals in the gathered `K` block (32 KiB of `f64`).
const CAP: usize = 4096;

/// Real lanes per accumulator in the portable fallback, used when the ISA has
/// no [`crate::kernel::DotFn`]; also the shallowest `K` block.
const W: usize = 4;

/// Shallowest `k` taken at each gathered extent, fitted on an Apple M5 against
/// the packed and direct paths (`docs/notebook/small-contractions.md`): the
/// register tile (16 rows for `f64`, 4 for complex) is mostly padding below
/// these extents. `(largest gn, smallest k)`, first match wins.
const REAL_RULE: [(usize, usize); 2] = [(4, 16), (8, 64)];
const CPLX_RULE: [(usize, usize); 2] = [(1, 16), (3, 32)];

fn unit_ramp(offs: &[i64]) -> bool {
    offs.windows(2).all(|w| w[1] - w[0] == 1)
}

/// Which operand is streamed: `Some(true)` for `B`, `Some(false)` for `A`.
/// `B` when both qualify and its free extent is the larger.
fn streamed_b(plan: &Plan) -> Option<bool> {
    let (m, n) = (plan.stats.m, plan.stats.n);
    match (unit_ramp(&plan.a_k), unit_ramp(&plan.b_k)) {
        (true, true) => Some(n >= m),
        (false, true) => Some(true),
        (true, false) => Some(false),
        (false, false) => None,
    }
}

/// An explicit [`Plan::with_blocking`] keeps the packed path.
pub(crate) fn use_kdot<T: Element>(plan: &Plan) -> bool {
    let s = &plan.stats;
    if plan.blocking.is_some() || s.k == 0 || s.m == 0 || s.n == 0 {
        return false;
    }
    let Some(sb) = streamed_b(plan) else {
        return false;
    };
    let gn = if sb { s.m } else { s.n };
    let per = if T::IS_COMPLEX { 4 } else { 1 };
    gn * per * W <= CAP && rule::<T>(gn, s.k)
}

fn rule<T: Element>(gn: usize, k: usize) -> bool {
    let table = if T::IS_COMPLEX {
        &CPLX_RULE
    } else {
        &REAL_RULE
    };
    table
        .iter()
        .find(|&&(g, _)| gn <= g)
        .is_some_and(|&(_, kmin)| k >= kmin)
}

/// # Safety
///
/// As [`crate::driver::execute`]; additionally [`use_kdot`] must have returned
/// true for `plan` and `T`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn execute<T>(
    plan: &Plan,
    alpha: T,
    a: *const T,
    b: *const T,
    beta: T,
    c: *const T,
    d: *mut T,
) where
    T: Element,
    T::Real: KernelSet,
{
    let dot = crate::kernel::config_for_plan::<T>(plan).ukr.func_dot;
    let sb = streamed_b(plan).unwrap_or(true);
    let k = plan.stats.k;
    // `g*`: gathered operand (extent `gn`); `s*`: streamed operand (extent `sn`).
    let (g_ptr, g_x, g_k, g_conj, gh) = if sb {
        (a, &plan.a_m, &plan.a_k, plan.conj_a, &plan.h_a)
    } else {
        (b, &plan.b_n, &plan.b_k, plan.conj_b, &plan.h_b)
    };
    let (s_ptr, s_y, s_k, s_conj, sh) = if sb {
        (b, &plan.b_n, &plan.b_k, plan.conj_b, &plan.h_b)
    } else {
        (a, &plan.a_m, &plan.a_k, plan.conj_a, &plan.h_a)
    };
    let (d_x, d_y, c_x, c_y) = if sb {
        (&plan.d_m, &plan.d_n, &plan.c_m, &plan.c_n)
    } else {
        (&plan.d_n, &plan.d_m, &plan.c_n, &plan.c_m)
    };
    let (gn, sn) = (g_x.len(), s_y.len());

    // `sum op_s(s) * op_g(g) == conj(sum s * conj(op_g(g)))` when `op_s`
    // conjugates, so the streamed side is read as stored.
    let g_flip = g_conj ^ s_conj;
    let out_conj = s_conj;

    let cplx = T::IS_COMPLEX;
    // Reals per gathered element: the value, plus its pair-swapped copy for
    // complex (gives the imaginary part without a shuffle).
    let per = if cplx { 4 } else { 1 };
    let kb = (CAP / (gn * per)).max(W).min(k);
    let read_c = beta != T::zero();
    let mut g_s = [MaybeUninit::<T::Real>::uninit(); CAP + 4 * W];
    let z = T::Real::ZERO;
    let g = {
        let len = (gn * kb * per).min(g_s.len());
        for s in g_s[..len].iter_mut() {
            s.write(z);
        }
        // SAFETY: the prefix was just initialised.
        unsafe { &mut *(&mut g_s[..len] as *mut [MaybeUninit<T::Real>] as *mut [T::Real]) }
    };

    for h in 0..plan.stats.batch {
        let (g0, s0) = (gh[h] as isize, sh[h] as isize);
        let ch = plan.h_c.get(h).copied().unwrap_or(0) as isize;
        let dh = plan.h_d[h] as isize;
        let mut p0 = 0;
        while p0 < k {
            let len = kb.min(k - p0);
            let lr = if cplx { 2 * len } else { len };
            // Gathered row `x`: `[x*kb*per ..]`, value then (complex) swapped copy.
            for x in 0..gn {
                let gx = g_x[x];
                let row = &mut g[x * kb * per..];
                for p in 0..len {
                    let v = *g_ptr.offset(g0 + (gx + g_k[p0 + p]) as isize);
                    let (re, im) = (v.re(), if g_flip { -v.im() } else { v.im() });
                    if cplx {
                        row[2 * p] = re;
                        row[2 * p + 1] = im;
                        row[lr + 2 * p] = im;
                        row[lr + 2 * p + 1] = re;
                    } else {
                        row[p] = re;
                    }
                }
            }
            let first = p0 == 0;
            let sk0 = s_k[p0];
            let mut y0 = 0;
            while y0 < sn {
                let ny = NB.min(sn - y0);
                // Tail rows alias the last valid one: computed, never stored.
                let mut srow = [core::ptr::null::<T::Real>(); NB];
                for (j, r) in srow.iter_mut().enumerate() {
                    let y = y0 + j.min(ny - 1);
                    *r = s_ptr.offset(s0 + (s_y[y] + sk0) as isize) as *const T::Real;
                }
                for (x, &dx) in d_x.iter().enumerate() {
                    let gr = g.as_ptr().add(x * kb * per);
                    let sums = match dot {
                        Some(f) => {
                            let mut o = [z; 2 * NB];
                            f(lr, srow.as_ptr(), gr, gr.add(lr), o.as_mut_ptr());
                            core::array::from_fn(|j| (o[2 * j], o[2 * j + 1]))
                        }
                        None if cplx => dots_cplx(srow, gr, gr.add(lr), lr),
                        None => dots_real(srow, gr, lr),
                    };
                    for (j, &(re, im)) in sums.iter().enumerate().take(ny) {
                        let y = y0 + j;
                        let s = T::from_parts(re, im);
                        let s = alpha.mul(if out_conj { s.conj() } else { s });
                        let doff = dh + (dx + d_y[y]) as isize;
                        let mut val = if first {
                            if read_c {
                                let coff = ch
                                    + (c_x.get(x).copied().unwrap_or(0)
                                        + c_y.get(y).copied().unwrap_or(0))
                                        as isize;
                                let cv = *c.offset(coff);
                                let cv = if plan.conj_c { cv.conj() } else { cv };
                                s.add(beta.mul(cv))
                            } else {
                                s
                            }
                        } else {
                            let cur = *d.offset(doff);
                            (if plan.conj_d { cur.conj() } else { cur }).add(s)
                        };
                        if plan.conj_d {
                            val = val.conj();
                        }
                        *d.offset(doff) = val;
                    }
                }
                y0 += NB;
            }
            p0 += len;
        }
    }
}

/// `NB` real dot products of length `lr` against one gathered row.
#[inline(always)]
unsafe fn dots_real<R: Real>(s: [*const R; NB], g: *const R, lr: usize) -> [(R, R); NB] {
    let z = R::ZERO;
    let mut acc = [[z; W]; NB];
    let main = lr / W * W;
    let mut t = 0;
    while t < main {
        let gv: [R; W] = core::array::from_fn(|l| *g.add(t + l));
        for j in 0..NB {
            for l in 0..W {
                acc[j][l] = (*s[j].add(t + l)).mul_add(gv[l], acc[j][l]);
            }
        }
        t += W;
    }
    for (j, a) in acc.iter_mut().enumerate() {
        for (l, v) in a.iter_mut().enumerate().take(lr - main) {
            *v = (*s[j].add(t + l)).mul_add(*g.add(t + l), *v);
        }
    }
    core::array::from_fn(|j| {
        let a = &acc[j];
        ((a[0] + a[1]) + (a[2] + a[3]), z)
    })
}

/// `NB` complex dot products over `lr` interleaved reals: `P1 += s .* g`
/// gives `re` as the alternating lane sum, `P2 += s .* swap(g)` gives `im`.
#[inline(always)]
unsafe fn dots_cplx<R: Real>(
    s: [*const R; NB],
    g: *const R,
    gs: *const R,
    lr: usize,
) -> [(R, R); NB] {
    let z = R::ZERO;
    let mut p1 = [[z; W]; NB];
    let mut p2 = [[z; W]; NB];
    let main = lr / W * W;
    let mut t = 0;
    while t < main {
        let gv: [R; W] = core::array::from_fn(|l| *g.add(t + l));
        let gw: [R; W] = core::array::from_fn(|l| *gs.add(t + l));
        for j in 0..NB {
            for l in 0..W {
                let sv = *s[j].add(t + l);
                p1[j][l] = sv.mul_add(gv[l], p1[j][l]);
                p2[j][l] = sv.mul_add(gw[l], p2[j][l]);
            }
        }
        t += W;
    }
    // `main` is a multiple of `W` (even), so lane parity is the element parity.
    for l in 0..lr - main {
        for j in 0..NB {
            let sv = *s[j].add(t + l);
            p1[j][l] = sv.mul_add(*g.add(t + l), p1[j][l]);
            p2[j][l] = sv.mul_add(*gs.add(t + l), p2[j][l]);
        }
    }
    core::array::from_fn(|j| {
        let (a, b) = (&p1[j], &p2[j]);
        ((a[0] - a[1]) + (a[2] - a[3]), (b[0] + b[1]) + (b[2] + b[3]))
    })
}
