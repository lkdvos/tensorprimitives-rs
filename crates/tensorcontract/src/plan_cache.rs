//! Per-thread cache of plans for [`crate::contract`].
//!
//! Keyed by everything [`Plan::new`] reads, so a hit is the plan a miss would
//! build. Lookup compares the borrowed operands against each key in place and
//! allocates nothing. Most recently used first; the capacity is shared by all
//! threads and set with [`set_plan_cache_capacity`].

use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::cell::RefCell;
use std::rc::Rc;
use std::vec::Vec;

use crate::error::Result;
use crate::plan::{Operand, Plan};

static CAPACITY: AtomicUsize = AtomicUsize::new(16);

/// Set how many plans each thread keeps for [`crate::contract`]; 0 disables
/// the cache. Default 16. Lookup is a linear scan, so keep it small.
pub fn set_plan_cache_capacity(plans: usize) {
    CAPACITY.store(plans, Relaxed);
}

/// The current per-thread plan cache capacity.
pub fn plan_cache_capacity() -> usize {
    CAPACITY.load(Relaxed)
}

thread_local! {
    static PLANS: RefCell<Vec<(Vec<i64>, Rc<Plan>)>> = const { RefCell::new(Vec::new()) };
}

/// The four operands in key order, `C` optional.
type Ops<'a> = [Option<Operand<'a>>; 4];

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

/// Whether `key` is `encode(ops)`, without building it.
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

/// Errors from [`Plan::new`] are returned and not cached.
pub(crate) fn plan_for(
    a: Operand<'_>,
    b: Operand<'_>,
    c: Option<Operand<'_>>,
    d: Operand<'_>,
) -> Result<Rc<Plan>> {
    let cap = plan_cache_capacity();
    let ops: Ops<'_> = [Some(a), Some(b), c, Some(d)];
    let hit = PLANS.with(|p| {
        let mut p = p.borrow_mut();
        p.truncate(cap);
        let i = p.iter().position(|(k, _)| matches(k, &ops))?;
        p[..=i].rotate_right(1);
        Some(Rc::clone(&p[0].1))
    });
    if let Some(plan) = hit {
        return Ok(plan);
    }
    let plan = Rc::new(Plan::new(a, b, c, d)?);
    if cap > 0 {
        PLANS.with(|p| {
            let mut p = p.borrow_mut();
            p.insert(0, (encode(&ops), Rc::clone(&plan)));
            p.truncate(cap);
        });
    }
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

    /// More shapes than the cache holds, cycled twice so every entry is
    /// evicted and rebuilt.
    #[test]
    fn contract_through_cache_matches_fresh_plans() {
        use crate::{contract, TensorView, TensorViewMut};
        let (i, j, p) = (b'i' as i64, b'j' as i64, b'p' as i64);
        for round in 0..2 {
            for s in 1..=(plan_cache_capacity() + 4) {
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

    #[test]
    fn zero_capacity_builds_every_call() {
        let idx = [b'i' as i64];
        let (l, ld) = (Layout::col_major(&[3]), Layout::col_major(&[]));
        let (a, b, d) = (
            Operand::new(&l, &idx),
            Operand::new(&l, &idx),
            Operand::new(&ld, &[]),
        );
        let prev = plan_cache_capacity();
        set_plan_cache_capacity(0);
        let (x, y) = (
            plan_for(a, b, None, d).unwrap(),
            plan_for(a, b, None, d).unwrap(),
        );
        set_plan_cache_capacity(prev);
        assert!(!Rc::ptr_eq(&x, &y));
        let (x, y) = (
            plan_for(a, b, None, d).unwrap(),
            plan_for(a, b, None, d).unwrap(),
        );
        assert!(Rc::ptr_eq(&x, &y));
    }
}
