// Reused-plan run time: MPS gates (uvst,lstc->luvc, physical dim d) and col-major GEMMs.
use num_complex::Complex64 as C;
use std::hint::black_box;
use std::time::Instant;
use tensorcontract::plan::Operand;
use tensorcontract::{Element, Layout, Plan, TensorView, TensorViewMut, parse_einsum};

fn time<F: FnMut()>(macs: usize, mut f: F) -> f64 {
    let iters = (20_000_000 / macs.max(1)).clamp(20, 200_000);
    for _ in 0..iters / 10 + 1 { f(); }
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..iters { f(); }
        best = best.min(t.elapsed().as_secs_f64() * 1e9 / iters as f64);
    }
    best
}

fn case<T: Element>(tag: &str, ea: &str, la: &Layout, lb: &Layout, ld: &Layout, mk: impl Fn(usize) -> T)
where T::Real: tensorcontract::kernel::KernelSet {
    let (ia, ib, id) = parse_einsum(ea).unwrap();
    let a: Vec<T> = (0..la.storage_len() as usize).map(&mk).collect();
    let b: Vec<T> = (0..lb.storage_len() as usize).map(&mk).collect();
    let mut d = vec![T::zero(); ld.storage_len() as usize];
    let plan = Plan::new(Operand::new(la, &ia), Operand::new(lb, &ib), None, Operand::new(ld, &id)).unwrap();
    let s = plan.stats.clone();
    let (mr, nr, _) = tensorcontract::kernel::plan_config::<T>(&plan);
    let ns = time(s.m * s.n * s.k * s.batch, || {
        plan.run(T::one(), TensorView::new(&a, la, &ia), TensorView::new(&b, lb, &ib), T::zero(), None,
                 TensorViewMut::new(&mut d, ld, &id)).unwrap();
        black_box(&d);
    });
    println!("{tag:<30} m={:<5} n={:<5} k={:<4} mr={mr} nr={nr} swap={} {ns:>10.0} ns",
             s.m, s.n, s.k, plan.transposes_gemm(mr) as u8);
}

fn main() {
    let c = |t: usize| C::new(1.0 + (t % 7) as f64, (t % 5) as f64);
    let f = |t: usize| 1.0 + (t % 7) as f64;
    let filter: Option<String> = std::env::args().nth(1);
    let want = |t: &str| filter.as_deref().map_or(true, |f| t.contains(f));
    for d in [2i64, 3, 4] {
        for l in [8i64, 16, 32, 64] {
            let tag = format!("gate d={d} l={l}");
            if !want(&tag) { continue; }
            let (lg, lt) = (Layout::col_major(&[d; 4]), Layout::col_major(&[l, d, d, l]));
            case::<C>(&tag, "uvst,lstc->luvc", &lg, &lt, &lt, c);
        }
    }
    for (m, n, k) in [(64i64, 512i64, 512i64), (256, 256, 1024), (100, 500, 600), (37, 4096, 300), (256, 1024, 2048), (1000, 1000, 1000)] {
        let tag = format!("deep f64 m={m} n={n} k={k}");
        if want(&tag) {
            let (la, lb, ld) = (Layout::col_major(&[m, k]), Layout::col_major(&[k, n]), Layout::col_major(&[m, n]));
            case::<f64>(&tag, "ip,pj->ij", &la, &lb, &ld, f);
        }
    }
    for (ty, cplx) in [("c64", true), ("f64", false)] {
        for m in [16i64, 64, 256] { for n in [64i64, 512, 4096] { for k in [4i64, 8, 16, 32, 64, 256] {
            let tag = format!("wb {ty} m={m} n={n} k={k}");
            if !want(&tag) { continue; }
            let (la, lb, ld) = (Layout::col_major(&[m, k]), Layout::col_major(&[k, n]), Layout::col_major(&[m, n]));
            if cplx { case::<C>(&tag, "ip,pj->ij", &la, &lb, &ld, c) } else { case::<f64>(&tag, "ip,pj->ij", &la, &lb, &ld, f) }
        }}}
        for bt in [false, true] {
        for m in [4i64, 8, 16, 32, 64, 128, 256] {
            for n in [256i64, 1024, 4096] {
                for k in [16i64, 64, 256] {
                    let tag = format!("gemm{} {ty} m={m} n={n} k={k}", if bt { "T" } else { "" });
                    if !want(&tag) { continue; }
                    let (la, ld) = (Layout::col_major(&[m, k]), Layout::col_major(&[m, n]));
                    let (lb, eb) = if bt { (Layout::col_major(&[n, k]), "ip,jp->ij") } else { (Layout::col_major(&[k, n]), "ip,pj->ij") };
                    if cplx { case::<C>(&tag, eb, &la, &lb, &ld, c) } else { case::<f64>(&tag, eb, &la, &lb, &ld, f) }
                }
            }
        }
        }
    }
    for (ty, cplx) in [("c64", true), ("f64", false)] {
        // M = 1 / N = 1 ("dot"): the vector times a matrix whose K is unit-stride or not.
        for (k, n) in [(64i64, 512i64), (256, 4096), (16, 4096), (1024, 256), (4096, 16)] {
            for (form, eb, lb) in [("Kunit", "p,pj->j", Layout::col_major(&[k, n])), ("Nunit", "p,jp->j", Layout::col_major(&[n, k]))] {
                let tag = format!("dot {ty} {form} k={k} n={n}");
                if !want(&tag) { continue; }
                let (la, ld) = (Layout::col_major(&[k]), Layout::col_major(&[n]));
                if cplx { case::<C>(&tag, eb, &la, &lb, &ld, c) } else { case::<f64>(&tag, eb, &la, &lb, &ld, f) }
            }
        }
        // Small M, N; deep K.
        for mn in [4i64, 8, 16, 32] { for k in [512i64, 4096] {
            for (form, ea, la) in [("Munit", "ip,pj->ij", Layout::col_major(&[mn, k])), ("Kunit", "pi,pj->ij", Layout::col_major(&[k, mn]))] {
                let tag = format!("deepk {ty} {form} mn={mn} k={k}");
                if !want(&tag) { continue; }
                let (lb, ld) = (Layout::col_major(&[k, mn]), Layout::col_major(&[mn, mn]));
                if cplx { case::<C>(&tag, ea, &la, &lb, &ld, c) } else { case::<f64>(&tag, ea, &la, &lb, &ld, f) }
            }
        }}
    }
    for (ty, cplx) in [("c64", true), ("f64", false)] {
        for m in [4i64, 8, 16, 32, 64, 128] { for n in [4i64, 16, 64, 512, 4096] { for k in [512i64, 2048, 8192] {
            if m * n * k > 1 << 30 { continue; }
            let tag = format!("kc {ty} m={m} n={n} k={k}");
            if !want(&tag) { continue; }
            let (la, lb, ld) = (Layout::col_major(&[m, k]), Layout::col_major(&[k, n]), Layout::col_major(&[m, n]));
            if cplx { case::<C>(&tag, "ip,pj->ij", &la, &lb, &ld, c) } else { case::<f64>(&tag, "ip,pj->ij", &la, &lb, &ld, f) }
        }}}
    }
    for (ty, cplx) in [("c64", true), ("f64", false)] {
        for m in [1i64, 2, 3, 4, 8, 12] { for n in [16i64, 512] { for k in [16i64, 32, 64, 128, 512] {
            let tag = format!("dotg {ty} m={m} n={n} k={k}");
            if !want(&tag) { continue; }
            let (la, lb, ld) = (Layout::col_major(&[m, k]), Layout::col_major(&[k, n]), Layout::col_major(&[m, n]));
            if cplx { case::<C>(&tag, "ip,pj->ij", &la, &lb, &ld, c) } else { case::<f64>(&tag, "ip,pj->ij", &la, &lb, &ld, f) }
        }}}
    }
    for (mn, k) in [(16i64, 32768i64), (32, 32768), (16, 4096), (32, 4096)] {
        for packed in [false, true] {
            let tag = format!("ubtest f64 mn={mn} k={k} packedB={}", packed as u8);
            if !want(&tag) { continue; }
            let la = Layout::col_major(&[mn, k]);
            let lb = if packed { Layout::new(vec![k, mn], vec![2, 2 * k]).unwrap() } else { Layout::col_major(&[k, mn]) };
            let ld = Layout::col_major(&[mn, mn]);
            case::<f64>(&tag, "ip,pj->ij", &la, &lb, &ld, f);
        }
    }
    for (ty, cplx, m) in [("f64", false, 32i64), ("f64", false, 24), ("f64", false, 20), ("f64", false, 28), ("c64", true, 8), ("c64", true, 6)] {
        for n in [16i64, 64, 512] { for k in [64i64, 256, 1024, 1536, 2048, 3072, 4096, 32768] {
            if m * n * k > 1 << 28 { continue; }
            let tag = format!("ubp {ty} m={m} n={n} k={k}");
            if !want(&tag) { continue; }
            let (la, lb, ld) = (Layout::col_major(&[m, k]), Layout::col_major(&[k, n]), Layout::col_major(&[m, n]));
            if cplx { case::<C>(&tag, "ip,pj->ij", &la, &lb, &ld, c) } else { case::<f64>(&tag, "ip,pj->ij", &la, &lb, &ld, f) }
        }}
    }
}
