//! Scatter and block-scatter vectors.
//!
//! A tensor participating in a contraction is viewed as a matrix whose rows
//! and columns are mixed-radix multi-indices over a *class* of index labels
//! (see [`crate::plan`]). The row scatter vector `rscat[i]` gives the element
//! offset of matrix row `i` relative to the tensor base; likewise `cscat[j]`
//! for columns. Element `(i, j)` then lives at `base + rscat[i] + cscat[j]`.
//!
//! The *block* scatter vector records, for each aligned block of `blk`
//! consecutive scatter entries, whether those entries form an arithmetic
//! progression, and with what stride. When they do, the corresponding slice of
//! the matrix can be traversed with ordinary strided (often unit-stride)
//! accesses — the "regular" fast path. When they do not, the generic gather
//! path is used. This is the block-scatter-matrix layout of Matthews
//! (arXiv:1607.00291).
//!
//! Note that a *zero* block stride is meaningful and legal here: it arises for
//! contraction indices that are absent from one operand (TAPP "isolated"
//! indices, i.e. reductions), where the same element is read repeatedly.
//! Irregular blocks are therefore flagged with a dedicated sentinel rather
//! than with zero as in some other implementations.
//!
//! Everything public here is a pure function of `i64` slices with no coupling
//! to the rest of the engine, which is why it is API rather than an internal
//! detail: it is the only way for a caller — the project's own benchmark
//! harness included — to describe the traversal the engine is about to make,
//! and to report regularity at the same granularity the engine sees. Combine
//! with [`crate::Plan::oriented_scatters`] and
//! [`crate::kernel::plan_config`], which give the vectors and the block sizes
//! actually used.
//!
//! # Forcing the gather path
//!
//! Both packing and the output write-back take the regular path when the block
//! scatter permits it. Setting `TENSORCONTRACT_WRITEBACK=gather` makes the
//! write-back use the general scatter loop unconditionally, disabling the
//! block-scatter row addressing and the `alpha = 1, beta = 0` copy along with
//! it. Results are unaffected — this exists so the fast path is an A/B switch at
//! run time rather than a rebuild, in the same spirit as
//! `TENSORCONTRACT_KERNEL=scalar` (see [`crate::kernel`]). Read once per
//! process.

/// Sentinel stored in a block-scatter vector for a block whose scatter entries
/// are *not* an arithmetic progression.
pub const IRREGULAR: i64 = i64::MIN;

/// Build the scatter vector for a mixed-radix index group.
///
/// `extents`/`strides` are ordered fastest-varying first. The returned vector
/// has length `extents.iter().product()` (1 for an empty group, i.e. a single
/// zero offset).
///
/// # Panics
///
/// Panics if `extents` and `strides` have different lengths — but only as a
/// `debug_assert_eq!`, because inside the crate both always come from one
/// [`crate::Layout`]. In a release build a *short* `strides` still panics, on
/// the index rather than the assertion, while a long one has its tail silently
/// ignored; a caller assembling the two slices separately should check the
/// lengths itself.
///
/// ```
/// use tensorcontract::scatter::build_scatter;
///
/// // Two modes: extent 2 stride 1 (fastest), extent 3 stride 10.
/// // Entry n is the element offset of the n'th index tuple.
/// assert_eq!(build_scatter(&[2, 3], &[1, 10]), vec![0, 1, 10, 11, 20, 21]);
///
/// // An empty group is one element at offset zero, not zero elements.
/// assert_eq!(build_scatter(&[], &[]), vec![0]);
/// ```
pub fn build_scatter(extents: &[i64], strides: &[i64]) -> Vec<i64> {
    debug_assert_eq!(extents.len(), strides.len());
    let total: i64 = extents.iter().product();
    let total = total.max(0) as usize;
    let mut out = Vec::with_capacity(total);
    if total == 0 {
        return out;
    }
    if extents.is_empty() {
        out.push(0);
        return out;
    }

    // The innermost axis is a tight contiguous loop; the odometer only runs
    // over the outer axes, once per inner run.
    let n0 = extents[0] as usize;
    let s0 = strides[0];
    let outer = extents.len() - 1;
    let mut small = [0i64; 8];
    let mut heap;
    let counter: &mut [i64] = if outer <= small.len() {
        &mut small[..outer]
    } else {
        heap = vec![0i64; outer];
        &mut heap
    };
    let mut base = 0i64;
    for _ in 0..total / n0 {
        let mut o = base;
        for _ in 0..n0 {
            out.push(o);
            o = o.wrapping_add(s0);
        }
        for d in 0..outer {
            let (e, s) = (extents[d + 1], strides[d + 1]);
            counter[d] += 1;
            base += s;
            if counter[d] < e {
                break;
            }
            base -= s * e;
            counter[d] = 0;
        }
    }
    debug_assert_eq!(out.len(), total);
    out
}

/// Derive the block-scatter vector for `scat` at block size `blk`.
///
/// Entry `b` covers `scat[b*blk .. min((b+1)*blk, len)]` and is the common
/// difference of that run, or [`IRREGULAR`]. Runs of length 0 or 1 are
/// trivially regular and report a stride of 0.
///
/// This is where the register block earns or loses the fast path: the *same*
/// scatter vector is fully regular at one block size and fully irregular at
/// another, which is why `MR` is a performance decision and not just a tile
/// shape.
///
/// # Panics
///
/// Panics if `blk` is 0. There is no block structure to describe at a block
/// size of zero, and the block count below would divide by it.
///
/// ```
/// use tensorcontract::scatter::{build_block_scatter, build_scatter, IRREGULAR};
///
/// let scat = build_scatter(&[2, 3], &[1, 10]);   // [0, 1, 10, 11, 20, 21]
///
/// // At blk = 2 every block sits inside one run of stride 1: all strided loads.
/// assert_eq!(build_block_scatter(&scat, 2), vec![1, 1, 1]);
///
/// // At blk = 3 every block straddles a run boundary: all gathers.
/// assert_eq!(build_block_scatter(&scat, 3), vec![IRREGULAR, IRREGULAR]);
/// ```
pub fn build_block_scatter(scat: &[i64], blk: usize) -> Vec<i64> {
    assert!(blk > 0);
    let nblk = scat.len().div_ceil(blk);
    let mut out = Vec::with_capacity(nblk);
    for b in 0..nblk {
        let lo = b * blk;
        let hi = (lo + blk).min(scat.len());
        out.push(run_stride(&scat[lo..hi]));
    }
    out
}

/// The block-scatter vectors one execution needs, built for a fixed register
/// block and orientation.
///
/// `c_m` is built on first use only: it is consulted just when `beta` is
/// nonzero, and the common `beta == 0` call never pays for it.
#[derive(Clone, Debug)]
pub(crate) struct BlockScatterSet {
    pub a_m: Vec<i64>,
    pub b_n: Vec<i64>,
    pub d_m: Vec<i64>,
    c_m: std::sync::OnceLock<Vec<i64>>,
}

impl BlockScatterSet {
    /// `am`, `bn`, `dm` are the oriented row scatters of the row operand, the
    /// column operand and the output.
    pub fn build(am: &[i64], bn: &[i64], dm: &[i64], mr: usize, nr: usize) -> Self {
        BlockScatterSet {
            a_m: build_block_scatter(am, mr),
            b_n: build_block_scatter(bn, nr),
            d_m: build_block_scatter(dm, mr),
            c_m: std::sync::OnceLock::new(),
        }
    }

    /// `C`'s row block scatter, built on the first call.
    pub fn c_m(&self, cm: &[i64], mr: usize) -> &[i64] {
        self.c_m.get_or_init(|| build_block_scatter(cm, mr))
    }
}

/// A plan's cache of its [`BlockScatterSet`].
///
/// The set is a pure function of the plan's scatters and of `(mr, nr, swap)`,
/// and the plan's scatters never change after construction, so it can live in
/// the plan. It holds the **first** key a plan executes with and nothing else:
/// a plan is almost always run at one element type, and a second key (an `f32`
/// run of a plan first run as `c64`, say) just builds its own set per call, as
/// every run did before. That keeps the cache a single `OnceLock` -- no lock,
/// no map, `Send + Sync` for free -- at the price of the unusual case staying
/// at the old speed.
///
/// `TENSORCONTRACT_BSCACHE=off` bypasses it, for A/B against the rebuilt-per-run
/// behaviour.
#[derive(Clone, Debug, Default)]
pub(crate) struct BlockScatterCache(std::sync::OnceLock<((usize, usize, bool), BlockScatterSet)>);

impl BlockScatterCache {
    /// The set for `(mr, nr, swap)`: the cached one when the key matches,
    /// otherwise `fresh` holds a newly built one.
    pub fn get<'a>(
        &'a self,
        key: (usize, usize, bool),
        am: &[i64],
        bn: &[i64],
        dm: &[i64],
        fresh: &'a mut Option<BlockScatterSet>,
    ) -> &'a BlockScatterSet {
        if cache_enabled() {
            let (k, set) = self
                .0
                .get_or_init(|| (key, BlockScatterSet::build(am, bn, dm, key.0, key.1)));
            if *k == key {
                return set;
            }
        }
        fresh.insert(BlockScatterSet::build(am, bn, dm, key.0, key.1))
    }
}

/// `TENSORCONTRACT_BSCACHE=off|on` (default on), read once per process.
fn cache_enabled() -> bool {
    env_once!(bool, "TENSORCONTRACT_BSCACHE", true, |v: &str| !matches!(
        v,
        "off" | "0" | "fresh"
    ))
}

/// Common difference of a run, or [`IRREGULAR`].
#[inline]
fn run_stride(run: &[i64]) -> i64 {
    if run.len() <= 1 {
        return 0;
    }
    let s = run[1] - run[0];
    for w in run.windows(2) {
        if w[1] - w[0] != s {
            return IRREGULAR;
        }
    }
    s
}

/// The run structure of a scatter vector: `Some((len, stride))` when it is a
/// concatenation of equal-length *maximal* arithmetic runs, `None` otherwise.
///
/// This is the shape every output scatter in the corpus has — a leading axis of
/// extent `len` and stride `stride`, restarted by the outer axes — and it is
/// what makes the effect of a candidate block size computable in `O(len)`
/// rather than by rebuilding a block scatter per candidate. It is also the
/// quantity both the orientation rule and the row-block rule turn on, so it
/// lives here rather than in either of them.
///
/// ```
/// use tensorcontract::scatter::{build_scatter, run_structure};
///
/// // Three maximal runs of length 2, each of stride 1.
/// let scat = build_scatter(&[2, 3], &[1, 10]);
/// assert_eq!(run_structure(&scat), Some((2, 1)));
///
/// // One contiguous axis is a single run spanning everything.
/// assert_eq!(run_structure(&build_scatter(&[6], &[1])), Some((6, 1)));
/// ```
pub fn run_structure(scat: &[i64]) -> Option<(usize, i64)> {
    if scat.len() < 2 {
        return None;
    }
    let stride = scat[1] - scat[0];
    // The first discontinuity ends the first run; every later run must match.
    let len = scat
        .windows(2)
        .position(|w| w[1] - w[0] != stride)
        .map_or(scat.len(), |p| p + 1);
    if len < 2 || !scat.len().is_multiple_of(len) {
        return None;
    }
    for (i, w) in scat.windows(2).enumerate() {
        let at_boundary = (i + 1) % len == 0;
        if !at_boundary && w[1] - w[0] != stride {
            return None;
        }
        // A boundary that happens to continue the progression would mean the
        // runs are longer than measured, contradicting maximality.
        if at_boundary && w[1] - w[0] == stride {
            return None;
        }
    }
    Some((len, stride))
}

/// Fraction of aligned `blk`-blocks of a `total`-entry scatter that fall inside
/// a single run, given runs of `len` entries.
///
/// Exactly the fraction that reaches a strided rather than a gather traversal.
///
/// ```
/// use tensorcontract::scatter::unbroken_fraction;
///
/// // Six entries in runs of 2. Blocks of 2 align with the runs exactly.
/// assert_eq!(unbroken_fraction(6, 2, 2), 1.0);
/// // Blocks of 4 cannot: the first straddles a boundary, the second does not.
/// assert_eq!(unbroken_fraction(6, 2, 4), 0.5);
/// ```
pub fn unbroken_fraction(total: usize, len: usize, blk: usize) -> f64 {
    if total == 0 || blk == 0 || len == 0 {
        return 1.0;
    }
    let nblk = total.div_ceil(blk);
    let whole = (0..nblk)
        .filter(|b| {
            let lo = b * blk;
            let hi = (lo + blk).min(total) - 1;
            lo / len == hi / len
        })
        .count();
    whole as f64 / nblk as f64
}

/// Fraction of blocks in a block-scatter vector that are regular. Used for
/// diagnostics and for the planar-vs-TTGT dispatch heuristic.
///
/// ```
/// use tensorcontract::scatter::{regular_fraction, IRREGULAR};
///
/// assert_eq!(regular_fraction(&[1, 1, IRREGULAR, 1]), 0.75);
/// // A zero block stride is regular: it is how a reduction's repeated read
/// // appears, and only IRREGULAR means the gather path.
/// assert_eq!(regular_fraction(&[0, 0]), 1.0);
/// ```
pub fn regular_fraction(bs: &[i64]) -> f64 {
    if bs.is_empty() {
        return 1.0;
    }
    let reg = bs.iter().filter(|&&s| s != IRREGULAR).count();
    reg as f64 / bs.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The original element-at-a-time odometer, kept as an oracle.
    fn build_scatter_oracle(extents: &[i64], strides: &[i64]) -> Vec<i64> {
        let total: i64 = extents.iter().product();
        let total = total.max(0) as usize;
        let mut out = Vec::with_capacity(total);
        if total == 0 {
            return out;
        }
        if extents.is_empty() {
            out.push(0);
            return out;
        }
        let n = extents.len();
        let mut counter = vec![0i64; n];
        let mut offset = 0i64;
        for _ in 0..total {
            out.push(offset);
            for d in 0..n {
                counter[d] += 1;
                offset += strides[d];
                if counter[d] < extents[d] {
                    break;
                }
                offset -= strides[d] * extents[d];
                counter[d] = 0;
            }
        }
        out
    }

    #[test]
    fn build_scatter_matches_oracle() {
        // Small xorshift so the test needs no dependency.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        for _ in 0..3000 {
            let rank = rnd(12) as usize; // includes 0 and > 8 (heap counter)
            let mut ext: Vec<i64> = (0..rank).map(|_| rnd(4) as i64).collect();
            // Zero extents are rare enough to leave most cases non-empty,
            // but cap the total so rank 11 stays cheap.
            while ext.iter().product::<i64>() > 5000 {
                let i = rnd(rank as u64) as usize;
                ext[i] = 1.max(ext[i] / 2);
            }
            if rnd(5) != 0 {
                for e in &mut ext {
                    *e = (*e).max(1);
                }
            }
            let strides: Vec<i64> = (0..rank).map(|_| rnd(41) as i64 - 20).collect();
            assert_eq!(
                build_scatter(&ext, &strides),
                build_scatter_oracle(&ext, &strides),
                "extents {ext:?} strides {strides:?}"
            );
        }
        assert_eq!(build_scatter(&[], &[]), vec![0]);
        assert!(build_scatter(&[3, 0], &[1, 3]).is_empty());
        assert!(build_scatter(&[0, 3], &[1, 3]).is_empty());
    }

    #[test]
    fn scatter_column_major() {
        // extents (2,3) strides (1,2), fastest first -> 0,1,2,3,4,5
        let s = build_scatter(&[2, 3], &[1, 2]);
        assert_eq!(s, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn scatter_permuted() {
        // A 2x3 tensor stored row-major, viewed with the *second* index
        // fastest: extents (3,2) strides (1,3).
        let s = build_scatter(&[3, 2], &[1, 3]);
        assert_eq!(s, vec![0, 1, 2, 3, 4, 5]);
        // and with the first index fastest: extents (2,3) strides (3,1)
        let s = build_scatter(&[2, 3], &[3, 1]);
        assert_eq!(s, vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn scatter_empty_group_is_single_zero() {
        assert_eq!(build_scatter(&[], &[]), vec![0]);
    }

    #[test]
    fn block_scatter_regularity() {
        let s = build_scatter(&[2, 3], &[3, 1]); // 0,3,1,4,2,5
        let bs = build_block_scatter(&s, 2);
        assert_eq!(bs, vec![3, 3, 3]);
        let bs = build_block_scatter(&s, 3);
        assert_eq!(bs, vec![IRREGULAR, IRREGULAR]);
        assert_eq!(regular_fraction(&bs), 0.0);
    }

    #[test]
    fn zero_stride_is_regular_not_irregular() {
        let s = build_scatter(&[4], &[0]);
        assert_eq!(s, vec![0, 0, 0, 0]);
        let bs = build_block_scatter(&s, 2);
        assert_eq!(bs, vec![0, 0]);
        assert_eq!(regular_fraction(&bs), 1.0);
    }

    #[test]
    fn run_structure_recognises_equal_length_runs() {
        // Four contiguous runs of 24, restarted by an outer axis.
        assert_eq!(
            run_structure(&build_scatter(&[24, 4], &[1, 200])),
            Some((24, 1))
        );
        // A single unbroken run is the whole vector.
        assert_eq!(run_structure(&build_scatter(&[24], &[1])), Some((24, 1)));
        // Constant non-unit stride is still one run: no block size breaks it.
        assert_eq!(run_structure(&build_scatter(&[12], &[4])), Some((12, 4)));
        // Unequal runs have no uniform structure.
        assert_eq!(run_structure(&[0, 1, 2, 100, 101, 200]), None);
        assert_eq!(run_structure(&[7]), None);
    }

    #[test]
    fn unbroken_fraction_counts_straddling_blocks() {
        // 96 entries in runs of 24. Only block sizes that tile a run survive.
        assert_eq!(unbroken_fraction(96, 24, 24), 1.0);
        assert_eq!(unbroken_fraction(96, 24, 8), 1.0);
        // 16 into 24 straddles every second block of a 48-entry period.
        assert!((unbroken_fraction(96, 24, 16) - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(unbroken_fraction(96, 24, 32), 0.0);
        assert_eq!(unbroken_fraction(96, 24, 48), 0.0);
    }

    #[test]
    fn ragged_tail_block() {
        let s: Vec<i64> = (0..5).collect();
        let bs = build_block_scatter(&s, 2);
        assert_eq!(bs, vec![1, 1, 0]); // last block has a single entry
    }
}
