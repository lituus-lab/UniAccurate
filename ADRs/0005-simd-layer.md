<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0005: SIMD summation layer

- Status: Accepted
- Date: 2026-07-20
- Scope: UniAccurate `src/UniAccurate/simd.nim`, the umbrella, the C ABI, and
  the `-d:simd` build path

## Decision

A SIMD acceleration layer sits at vgraph index 2 (`twosum < algorithms < simd <
c_api`). It is gated by `-d:simd`: a scalar build (no flag) never imports
[nimsimd](https://github.com/lbartoletti/nimsimd), and `simd.nim` is an empty
module without it. nimsimd is external infrastructure, pinned in `requires` to
the `neon-avx512` fork branch (the way NimContracts is pinned), and is not
subject to `checkVGraph`.

The kernels keep `L` per-lane running sums in one vector accumulator, then
scalar-reduce the lanes — `storev` to a stack array, no horizontal-add
intrinsic. The compensated kernels (`kahan`/`neumaier`/`klein`) merge the `L`
lane partials with the scalar compensated recurrence from `algorithms/`: a
SIMD-block, scalar-merge scheme. `naive` and `pairwise` carry no reliability
flag; `pairwise` reuses `PairwiseThreshold` with a SIMD naive base case.

ISA selection is compile-time, by `defined(avx512)` / `defined(avx2)` /
`defined(arm64)` — nimsimd branches on the same defines. `config.nims` adds the matching `-mavx512f -mfma` / `-mavx2 -mfma` (or
`/arch:AVX512` / `/arch:AVX2` under MSVC, which implies FMA); arm64 needs
nothing, NEON-FMA is the base ISA. `-d:scalarUniAccurate` opts out.

**Dispatch lives at the top, not in the algorithms.** The vgraph forbids a
back-edge `algorithms → simd`, so unlike a flat library UniAccurate cannot
dispatch SIMD inside an algorithm module. The umbrella exports the `*Simd`
procs under `-d:simd` (Nim users opt in explicitly; the scalar `naiveSum` etc.
stay scalar), and the C ABI dispatches each `ua_sum_*` through the SIMD kernel
with a scalar fallback. The C header is unchanged (SIMD adds no `ua_*` symbols).

**Reliability and fallback.** A compensated SIMD result is `(T, bool)`:
`reliable = isFin(r) and maxLane <= LaneConcentrationFallback * abs(r)` with
`LaneConcentrationFallback = 1024.0`. When lane partials concentrate relative
to the result (cancellation), the caller falls back to the scalar algorithm,
which is bit-identical to the no-`-d:simd` build. `naive`/`pairwise` take the
bare SIMD result — the naive forward-error bound holds regardless of lane
concentration.

**float64 vs float32.** NEON has no float64 path, so `simdF64Enabled =
defined(avx2) or defined(avx512)` and the C ABI (float64-only) dispatches SIMD
only on amd64 AVX2/AVX-512. On arm64 the C ABI stays scalar even with
`-d:simd`; NEON is exercised through the float32 Nim tests
(`simdF32Enabled = simdF64Enabled or defined(arm64)`).

**No FMA in v1.** ADR-0004 forbids FMA inside the compensated recurrence, and
v1 has no dot product, so the layer uses no FMA at all. The dot-product kernels
that would use FMA are deferred to a later task; ADR-0004 still applies when
they arrive. `-ffp-contract=off` from `config.nims` is unchanged. *Superseded
by ADR-0007: the SIMD dot kernels (`naiveDotSimd`/`dot2Simd`/`dotK3Simd`) now
use FMA intrinsics; the compensated-recurrence FMA ban (ADR-0004) stands.*

## Invariants

1. `-d:simd` is the master gate; without it `simd.nim` is empty and nimsimd is
   never imported.
2. The algorithm modules (`naivesum`, `pairwisesum`, `compensatedsum`) never
   import `simd` — SIMD dispatch happens at the umbrella and C ABI only.
3. `simdF64Enabled` is false on arm64; the C ABI falls back to scalar there.
4. Compensated SIMD returns `(T, bool)` and the C ABI falls back to the scalar
   algorithm when `reliable = false`.
5. No FMA in the compensated-sum recurrence; the SIMD dot kernels (ADR-0007)
   use FMA intrinsics.
6. `-d:scalarUniAccurate` opts out of the FMA and `-mavx*` flags; do not pair it
   with `-d:simd` (the umbrella still imports `simd.nim` under `-d:simd` alone).
