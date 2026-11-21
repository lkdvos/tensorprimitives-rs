## Small contractions: the per-call floor, the direct path, and the plan cache

Part 23. Apple M5, single thread, no pinning (Darwin), 2026-11-20. **One machine,
so every threshold below is a per-microarchitecture claim (A56).** Other agents
were compiling on the same machine during parts of this session; every number
here is a within-session, interleaved A/B, and none should be quoted off this
machine.

### Why

An MPS downstream (`mps-rs`) profiled the two-site gate
`out[l,u,v,c] = sum g[u,v,s,t] th[l,s,t,c]` on qubits — `c64`, `m = 4`, `k = 4`,
`n = l^2` — and found a naive loop beating a *reused* plan at every bond
dimension: ~14x at `l = 1`, ~1.4x at `l = 19..64`. It bypassed the engine for
physical dimension `<= 16`. Its four findings, all confirmed against the code at
`8cda75e`:

1. `Blocking::derive` read `TENSORCONTRACT_KC_COUPLE` with an uncached `getenv`
   on every `Plan::run` (~25% of a tiny run).
2. Pack buffers were sized by the blocking `kc` (256 for `c64`), not
   `min(kc, k)`, and allocated on every run: 41 KB for a 64-byte problem.
3. No path that skips packing; at `l = 64` pack + write-back were ~60% of
   samples and the micro-kernel ~22%.
4. `build_scatter` was an O(size) odometer, so planning scaled with the problem
   (22 us at `l = 64`).

QuasiStrided.jl's history was read for prior art
(`quasistrided-ideas.md`, outside the repo): it tried a scalar direct path and
removed it (25–40x slower than packed SIMD at `k = 64`), tried unpacked `A` and
removed it (1.01–1.08x, AVX2 regressions at tiny sizes), and kept only unpacked
`B` under `M <= 256`. The lesson taken: a direct path must vectorise, and must
be confined by rule to the regime where packing cannot amortise.

### What changed

| change | switch | bitwise |
|---|---|---|
| `KC_COUPLE` read once (`env_once!`) | — | yes |
| pack panels sized by `min(kc, k)` (allocation only; the blocking is untouched, so this is not the refuted `min(k, KC)` re-derivation) | — | yes |
| `check_bounds` O(1) from extremes cached in the plan | — | yes |
| block-scatter tables cached in the plan (first `(MR, NR, orientation)` key) | — | yes |
| serial-path panels and tile from a per-thread scratch (`set_scratch_limit`, default 1 MiB; threaded path untouched, A55) | — | yes |
| `build_scatter` inner axis as a contiguous loop; run structure from folded axes; fewer allocations in `Plan::new` | — | yes |
| direct (unpacked) path, `direct.rs` | — | **no** |
| per-thread plan cache in `contract()` (`set_plan_cache_capacity`, default 16) | — | yes |

`config_for_plan` was measured at 13–14 ns per call and left uncached.

### The direct path, and how it got fast

Three versions were measured against the packed path on a 216-cell grid
(`m` 2..64, `n` 16..4096, `k` 1..64, `c64` and `f64`, `DIRECT=on` vs `off`):

1. **Gather the smaller block, dot product over `k`.** Lost broadly from
   `m = 16`: the dot product is over the one extent that is small, so it does
   not vectorise.
2. **Axpy across the gathered extent, accumulators in memory, 48 KB of stack
   zeroed per call.** Worse everywhere — the zeroing alone was visible at the
   sizes the path exists for.
3. **Gather the smaller-extent operand whole, stream the other in chunks of 8,
   accumulate each chunk in registers; uninitialised stack, only the used
   prefix written.** Wins wherever packing does not amortise — but `c64` still
   lost from `k ≈ 2`, because Rust does not contract `a*b + c` into an FMA, so
   each complex lane cost 8 ops where the packed kernel spends 4 FMAs.
4. **`Real::mul_add` (fused only where FMA is baseline: aarch64, or x86 with
   `+fma` — elsewhere `f64::mul_add` is a libm call) and the 4m form (four real
   accumulators, the subtraction once per output).** This is what shipped.

The selection rule was fitted on a second, 648-cell grid (`m` 1..64, `n`
8..4096, `k` 1..32) by scoring candidate rules offline against the per-cell
best arm. Packed-only is 28% (`c64`) and 30% (`f64`) off that oracle in
geomean; `macs <= 2048 || (min(m,n) <= 3 && k <= 32)` is 0.4% and 1.8% off.
`m = 4` is exactly the M5's `c64` register block and the packed kernel wins there
on column-major GEMMs from `n = 64` — but on the MPS gate itself, whose output
and `th` are strided along `N`, forcing the direct path is ~10% faster at `l = 19`
and `l = 64`. `MINDIM` therefore ships at 4, costing the GEMM grid ~1%.

The path is chosen before the kernel config is looked up, and **independently of
the thread count**: the first version gated on threads and broke
`threaded_two_d_barrier_stress`, because `THREADS` is promised bitwise identical
and this path rounds differently from the packed one.

### Results (MPS two-site gate, `c64`, `tc_small` from `mps-rs`)

Run time of a reused plan, then `contract()` (plan + run), ns. Baseline is
`8cda75e`; "final" is `main` after this part.

| `l` | run, baseline | run, final | `contract()`, baseline | `contract()`, final | naive loop |
|---|---|---|---|---|---|
| 1 | 251 | 41 | ~900–1600 | 60 | 22 |
| 4 | 461 | 153 | ~1300 | 170 | 171 |
| 19 | 3780 | 2160 | ~5800 | ~2250 | 2490 |
| 64 | 38900 | 19000 | ~59000 | ~19500 | 28300 |

Per-run heap allocations went from 6 (41–566 KB) to 0; `Plan::new` allocations
from 37–48 to 19, and its time at `l = 64` from ~19 us to ~3.7 us.

**Then the unit-stride chunks.** At `l = 19..64` the naive loop was still ~1.2x
ahead, and the cost was the per-element scatter-table gather of `th` and store of
`D` along `N`. In the MPS layout both are unit-stride along `l`, so a full
8-wide chunk whose offsets are consecutive is now loaded and stored as one run.
Reused-plan run: `l = 19` 2.81 → 2.16 us, `l = 64` 32.0 → 19.0 us — now ahead of
the naive loop (2.50 and 28.4 us) at every size. The rule was re-fitted on the
648-cell grid afterwards: `MACS` 2048 → 4096 (packed-only is now 38–39% off the
oracle, the rule ~2% / ~1%).

**What is still owed.** A second
machine class (x86, where the direct path does not fuse unless built with
`+fma`) has not been measured, so the rule's defaults are M5 numbers.

### Three negative results

Padding-aware row-block choice and an explicit lane-chunked packer both measured
as no win (`refuted.md`, part 23). The packer finding moves the target: the
existing packer already vectorises unit-stride blocks, and packing is ~5% of a
small-`k` run, so the remaining overhead there is write-back.

The third:

Hoisting the `TileFormat` dispatch out of the write-back's inner loop, on the
theory that a run-time `match` per element blocked vectorisation of the plane
recombination, measured within 1% (`c64`, `m` 16..256, `k` 16..256). LLVM had
already unswitched it. Not committed; recorded in `refuted.md`.

### Switches retired

On review the bitwise-identical changes (scratch reuse, block-scatter cache,
plan cache) ship without switches, and the two caches are sized in code
(`set_scratch_limit`, `set_plan_cache_capacity`). The direct path's thresholds
are constants, and the direct path has no switch either. An explicit
`Plan::with_blocking` takes the packed path, so the randomised sweeps (which run
every problem with a tiny explicit blocking and with none) still cover the packed
driver on small shapes. `threaded_clamps_below_one_cell_per_thread` uses a deeper
`k` so its small-`m` cases still reach the threaded driver, and
`direct_path_matches_the_oracle` covers the direct path's branches; a mutation in
each of its contiguous load, contiguous store and conjugation paths fails it.
