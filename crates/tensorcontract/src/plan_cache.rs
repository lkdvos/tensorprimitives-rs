//! A small per-thread cache of plans for [`crate::contract`].
//!
//! [`crate::contract`] is the one-shot entry point, so every call used to build
//! a [`Plan`] and throw it away. At the sizes an MPS gate loop works at, that
//! planning is most of the call: about 370 ns of planning around a 40 ns run
//! for a 64-byte two-site gate on an Apple M5. A caller in that regime almost
//! always repeats a handful of shapes, so the plans are kept.
//!
//! The key is everything [`Plan::new`] reads — each operand's extents, strides,
//! labels and element op, and whether `C` is present — so a hit is exactly the
//! plan a miss would build. Lookup compares the borrowed operands against each
//! stored key in place, without building a key, so a hit allocates nothing.
//! The cache is most-recently-used first and holds [`CAPACITY`] plans per
//! thread; it is per thread so there is no locking on the hot path.
//!
//! `TENSORCONTRACT_PLANCACHE=off` restores the build-every-call behaviour, so
//! both arms can be A/B'd in one process restart. A cached plan is the same
//! plan, so results are bitwise identical either way.

use std::cell::RefCell;
use std::rc::Rc;
use std::vec::Vec;

use crate::error::Result;
use crate::plan::{Operand, Plan};

/// Plans kept per thread. Small on purpose: lookup is a linear scan, and a
/// gate loop cycles through a few shapes, not hundreds.
const CAPACITY: usize = 16;

fn enabled() -> bool {
    env_once!(bool, "TENSORCONTRACT_PLANCACHE", true, |v: &str| !matches!(
        v.to_ascii_lowercase().as_str(),
        "off" | "0" | "false"
    ))
}

thread_local! {
    static PLANS: RefCell<Vec<(Vec<i64>, Rc<Plan>)>> = const { RefCell::new(Vec::new()) };
}

/// The four operands in key order, `C` optional.
type Ops<'a> = [Option<Operand<'a>>; 4];

/// Flatten the operands into a key. Only called on a miss.
fn encode(ops: &Ops<'_>) -> Vec<i64> {
    let mut key = Vec::new();
    for op in ops {
        match op {
            None => key.push(-1),
            Some(o) => {
                key.push(o.layout.ndim() as i64);
                key.push(o.op.is_conj() as i64);
                key.extend_from_slice(o.layout.extents());
                key.extend_from_slice(o.layout.strides());
                key.push(o.idx.len() as i64);
                key.extend_from_slice(o.idx);
            }
        }
    }
    key
}

/// Whether `key` is the encoding of `ops`, compared in place.
fn matches(key: &[i64], ops: &Ops<'_>) -> bool {
    let mut rest = key;
    let mut take = |n: usize| -> Option<&[i64]> {
        let (head, tail) = rest.split_at_checked(n)?;
        rest = tail;
        Some(head)
    };
    for op in ops {
        let ok = match op {
            None => take(1) == Some(&[-1][..]),
            Some(o) => {
                let nd = o.layout.ndim();
                take(2) == Some(&[nd as i64, o.op.is_conj() as i64][..])
                    && take(nd) == Some(o.layout.extents())
                    && take(nd) == Some(o.layout.strides())
                    && take(1) == Some(&[o.idx.len() as i64][..])
                    && take(o.idx.len()) == Some(o.idx)
            }
        };
        if !ok {
            return false;
        }
    }
    rest.is_empty()
}

/// The plan for these operands: from this thread's cache, or built and cached.
/// Errors from [`Plan::new`] are returned and not cached.
pub(crate) fn plan_for(
    a: Operand<'_>,
    b: Operand<'_>,
    c: Option<Operand<'_>>,
    d: Operand<'_>,
) -> Result<Rc<Plan>> {
    if !enabled() {
        return Plan::new(a, b, c, d).map(Rc::new);
    }
    let ops: Ops<'_> = [Some(a), Some(b), c, Some(d)];
    let hit = PLANS.with(|p| {
        let mut p = p.borrow_mut();
        let i = p.iter().position(|(k, _)| matches(k, &ops))?;
        // Move to the front so the scan finds the hot shapes first.
        p[..=i].rotate_right(1);
        Some(Rc::clone(&p[0].1))
    });
    if let Some(plan) = hit {
        return Ok(plan);
    }
    let plan = Rc::new(Plan::new(a, b, c, d)?);
    PLANS.with(|p| {
        let mut p = p.borrow_mut();
        p.insert(0, (encode(&ops), Rc::clone(&plan)));
        p.truncate(CAPACITY);
    });
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Layout;
    use crate::plan::ElementOp;

    #[test]
    fn key_round_trips_and_distinguishes() {
        let (l1, l2) = (Layout::col_major(&[2, 3]), Layout::col_major(&[3, 2]));
        let ops = |la: &'static Layout, conj: bool| -> Ops<'static> {
            let mut a = Operand::new(la, &[1, 3]);
            if conj {
                a.op = ElementOp::Conjugate;
            }
            [
                Some(a),
                Some(Operand::new(la, &[3, 2])),
                None,
                Some(Operand::new(la, &[1, 2])),
            ]
        };
        let l1: &'static Layout = Box::leak(Box::new(l1));
        let l2: &'static Layout = Box::leak(Box::new(l2));
        let base = ops(l1, false);
        let key = encode(&base);
        assert!(matches(&key, &base));
        assert!(!matches(&key, &ops(l2, false)), "different layout");
        assert!(!matches(&key, &ops(l1, true)), "different op");
        let mut with_c = base;
        with_c[2] = Some(Operand::new(l1, &[1, 2]));
        assert!(!matches(&key, &with_c), "C present vs absent");
        assert!(!matches(&key[..key.len() - 1], &base), "truncated key");
    }

    /// More shapes than the cache holds, each run twice in a cycle so every
    /// one is evicted and rebuilt, and some hit: every result must match a
    /// fresh plan's, data changing between calls.
    #[test]
    fn contract_through_cache_matches_fresh_plans() {
        use crate::{contract, TensorView, TensorViewMut};
        let (i, j, p) = (b'i' as i64, b'j' as i64, b'p' as i64);
        for round in 0..2 {
            for s in 1..=(CAPACITY + 4) {
                let (m, n, k) = (s, s % 5 + 1, s % 3 + 1);
                let la = Layout::col_major(&[m as i64, k as i64]);
                let lb = Layout::col_major(&[k as i64, n as i64]);
                let ld = Layout::col_major(&[m as i64, n as i64]);
                let a: Vec<f64> = (0..m * k).map(|x| (x + round) as f64).collect();
                let b: Vec<f64> = (0..k * n).map(|x| (x * 2 + s) as f64).collect();
                let mut got = vec![0.0; m * n];
                let mut want = vec![0.0; m * n];
                contract(
                    1.0,
                    TensorView::new(&a, &la, &[i, p]),
                    TensorView::new(&b, &lb, &[p, j]),
                    0.0,
                    None,
                    TensorViewMut::new(&mut got, &ld, &[i, j]),
                )
                .unwrap();
                Plan::new(
                    Operand::new(&la, &[i, p]),
                    Operand::new(&lb, &[p, j]),
                    None,
                    Operand::new(&ld, &[i, j]),
                )
                .unwrap()
                .run(
                    1.0,
                    TensorView::new(&a, &la, &[i, p]),
                    TensorView::new(&b, &lb, &[p, j]),
                    0.0,
                    None,
                    TensorViewMut::new(&mut want, &ld, &[i, j]),
                )
                .unwrap();
                assert_eq!(got, want, "shape {m}x{n}x{k} round {round}");
            }
        }
    }
}
