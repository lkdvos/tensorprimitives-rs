//! One engine per process: `xbench <tp|tf|tfc> cases.txt` prints `id ns_per_call checksum`.
use num_complex::Complex64 as C64;
use std::time::Instant;
use tenferro_cpu::CpuEngine;
use tenferro_tensor::{DotGeneralConfig, SessionCachedDot, Tensor, TensorDot};

trait Val: Copy + Default + tensorcontract::Element + tenferro_tensor::TensorScalar {
    fn gen(idx: usize, seed: usize) -> Self;
    fn sum(x: &[Self]) -> f64;
}
fn re(idx: usize, seed: usize) -> f64 { ((idx * 17 + seed * 13 + 3) % 97) as f64 / 97.0 - 0.5 }
fn im(idx: usize, seed: usize) -> f64 { ((idx * 29 + seed * 7 + 5) % 89) as f64 / 89.0 - 0.5 }
impl Val for f64 {
    fn gen(i: usize, s: usize) -> Self { re(i, s) }
    fn sum(x: &[Self]) -> f64 { x.iter().enumerate().map(|(i, v)| v * (1.0 + (i % 7) as f64)).sum() }
}
impl Val for C64 {
    fn gen(i: usize, s: usize) -> Self { C64::new(re(i, s), im(i, s)) }
    fn sum(x: &[Self]) -> f64 {
        x.iter().enumerate().map(|(i, v)| (v.re + 0.5 * v.im) * (1.0 + (i % 7) as f64)).sum()
    }
}

struct Case { id: String, dt: String, ca: bool, cb: bool, ia: Vec<u8>, ib: Vec<u8>, ic: Vec<u8>, ext: [usize; 128] }

fn parse(line: &str) -> Case {
    let f: Vec<&str> = line.split_whitespace().collect();
    let mut ext = [0usize; 128];
    for kv in f[7].split(',') {
        let (k, v) = kv.split_once('=').unwrap();
        ext[k.as_bytes()[0] as usize] = v.parse().unwrap();
    }
    let ic = if f[6] == "-" { vec![] } else { f[6].as_bytes().to_vec() };
    Case { id: f[0].into(), dt: f[1].into(), ca: f[2] == "1", cb: f[3] == "1",
           ia: f[4].as_bytes().to_vec(), ib: f[5].as_bytes().to_vec(), ic, ext }
}

/// Times `f` in batches of ~1 ms; median ns per call over `reps` batches.
fn time(mut f: impl FnMut()) -> f64 {
    for _ in 0..3 { f(); }
    let t = Instant::now(); f(); let one = t.elapsed().as_nanos().max(1) as f64;
    let n = ((1e6 / one) as usize).clamp(1, 200_000);
    let mut v: Vec<f64> = (0..15).map(|_| {
        let t = Instant::now();
        for _ in 0..n { f(); }
        t.elapsed().as_nanos() as f64 / n as f64
    }).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn run<T: Val>(c: &Case, eng: &str, tf: &mut CpuEngine) -> (f64, f64)
where <T as tensorcontract::Element>::Real: tensorcontract::KernelSet {
    let sh = |l: &[u8]| l.iter().map(|&x| c.ext[x as usize]).collect::<Vec<_>>();
    let (sa, sb, sc) = (sh(&c.ia), sh(&c.ib), sh(&c.ic));
    let n = |s: &[usize]| s.iter().product::<usize>();
    let a: Vec<T> = (0..n(&sa)).map(|i| T::gen(i, 1)).collect();
    let b: Vec<T> = (0..n(&sb)).map(|i| T::gen(i, 2)).collect();
    if eng == "tp" {
        use tensorcontract::{contract, Layout, TensorView, TensorViewMut};
        let lay = |s: &[usize]| Layout::col_major(&s.iter().map(|&x| x as i64).collect::<Vec<_>>());
        let lab = |l: &[u8]| l.iter().map(|&x| x as i64).collect::<Vec<_>>();
        let (la, lb, lc) = (lay(&sa), lay(&sb), lay(&sc));
        let (xa, xb, xc) = (lab(&c.ia), lab(&c.ib), lab(&c.ic));
        let mut d = vec![T::default(); n(&sc)];
        let mut go = |d: &mut [T]| {
            let mut va = TensorView::new(&a, &la, &xa);
            let mut vb = TensorView::new(&b, &lb, &xb);
            if c.ca { va = va.conj(); }
            if c.cb { vb = vb.conj(); }
            contract(<T as tensorcontract::Element>::one(), va, vb, <T as tensorcontract::Element>::zero(), None, TensorViewMut::new(d, &lc, &xc)).unwrap();
        };
        go(&mut d);
        let ck = T::sum(&d);
        (time(|| go(&mut d)), ck)
    } else {
        let pos = |l: &[u8], x: u8| l.iter().position(|&y| y == x).unwrap();
        let shared: Vec<u8> = c.ia.iter().copied().filter(|x| c.ib.contains(x)).collect();
        let free: Vec<u8> = c.ia.iter().chain(c.ib.iter()).copied().filter(|x| !shared.contains(x)).collect();
        assert_eq!(free, c.ic, "{}: output must be freeA ++ freeB", c.id);
        let cfg = DotGeneralConfig {
            lhs_contracting_dims: shared.iter().map(|&x| pos(&c.ia, x)).collect::<Vec<_>>().as_slice().into(),
            rhs_contracting_dims: shared.iter().map(|&x| pos(&c.ib, x)).collect::<Vec<_>>().as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
        };
        let ta = Tensor::from_vec_col_major(sa.clone(), a.clone()).unwrap();
        let tb = Tensor::from_vec_col_major(sb.clone(), b.clone()).unwrap();
        let cached = eng == "tfc";
        let mut go = |e: &mut CpuEngine| -> Tensor {
            if cached { e.dot_general_with_conj_cached(Some(0), &ta, &tb, &cfg, c.ca, c.cb).unwrap() }
            else { e.dot_general_with_conj(&ta, &tb, &cfg, c.ca, c.cb).unwrap() }
        };
        let r = go(tf);
        let ck = T::sum(r.as_slice::<T>().unwrap());
        (time(|| { std::hint::black_box(go(tf)); }), ck)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let eng = args[1].as_str();
    let filt = args.get(3).cloned().unwrap_or_default();
    let mut tf = CpuEngine::serial();
    for line in std::fs::read_to_string(&args[2]).unwrap().lines().filter(|l| !l.is_empty()) {
        let c = parse(line);
        if !c.id.contains(&filt) { continue; }
        let (ns, ck) = match c.dt.as_str() {
            "f64" => run::<f64>(&c, eng, &mut tf),
            _ => run::<C64>(&c, eng, &mut tf),
        };
        println!("{} {} {:.1} {:.10e}", c.id, c.dt, ns, ck);
    }
}
