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
the `lbartoletti/nimsimd` fork's `master` branch (the way NimContracts tracks
`lbartoletti/NimContracts#main`) -- the AVX-512/NEON extras that once lived on
a separate `neon-avx512` branch are on `master` now -- and is not subject to
`checkVGraph`.

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

**No FMA at v1 (historical).** ADR-0004 forbids FMA inside the compensated
recurrence, and v1 shipped without a dot product, so the layer used no FMA at
the time; the dot-product kernels that would use FMA were deferred. They
arrived in ADR-0007: the SIMD dot kernels (`naiveDotSimd`/`dot2Simd`/`dotK3Simd`)
now use FMA intrinsics, while the compensated-recurrence FMA ban (ADR-0004)
still stands. `-ffp-contract=off` from `config.nims` is unchanged.

**Amendment (ADR-0008).** `simd.nim` is no longer empty without `-d:simd`: the
dot-kernel templates (`defineNaiveDotV`/`defineDot2V`/`defineDotK3V`) and their
small dependency set (`./twosum`, `./algorithms/compensatedsum`,
`LaneConcentrationFallback`) sit unconditionally above the gate so
`simd_dispatch.nim` can reuse them for runtime ISA dispatch, independent of
`-d:simd`. This adds no new real dependency (both imports were already pulled
in unconditionally by the umbrella) and nimsimd is still never imported by this
path without either `-d:simd` or amd64 (`simd_dispatch.nim`'s own gate).
Invariant 1 below covers the sum family and the ISA instantiation blocks only.

**Amendment (ADR-0008, naive NaN recovery).** "`naive`/`pairwise` take the bare
SIMD result" holds for the forward-error bound but not for NaN: the L lane sums
accumulate independently, so on overflow-prone input one lane can reach `+Inf`
and another `-Inf` and the merge yields NaN, which the left-to-right scalar body
cannot reach (`Inf + finite = Inf`). `naiveSumSimd` therefore falls back to the
scalar body when its result is NaN and every input was finite -- keeping it
bit-identical to the no-`-d:simd` build, and matching the recovery `ua_dot_naive`
already applied around the naive dot kernel. The compensated kernels need no such
guard: a non-finite result is already `reliable = false`.

**Amendment (ADR-0008, reliability tuple type).** The `(T, bool)` pair is the
named type `SimdResult[T]`, not an anonymous tuple. A generic wrapper returning
`(T, bool)` at `T = float64` and a concrete kernel returning `(float64, bool)`
are the same Nim type, but the C backend emitted two distinct structs and
rejected the assignment, which kept the `-d:simd` C ABI build from compiling at
all on amd64.

**Amendment (ADR-0008, lane merge).** The dot kernels' scalar merge takes the
`L` lane sums and their `L` (resp. `2L`) compensation terms as separate
addends. Collapsing each lane first (`sl[j] + el[j]`, then merging the `L`
results) rounds the compensation away at ~`eps * max|sl|`, which exceeds the
twice/threefold bound on cancellation data while lane concentration stays
under `LaneConcentrationFallback` -- so the reliability check below does not
catch it. See ADR-0008's Verification for the measured case.

## Invariants

1. `-d:simd` is the master gate for the sum family (`naiveSumSimd`/
   `kahanSumSimd`/`neumaierSumSimd`/`kleinSumSimd`/`pairwiseSumSimd`) and for
   the compile-time-locked `naiveDotSimd`/`dot2Simd`/`dotK3Simd` wrappers;
   without it the ISA instantiation blocks (`when defined(avx512): ... elif
   defined(avx2):`) and the NEON block never compile, and nimsimd is not
   imported by this path. The dot-kernel templates above the gate are
   template definitions only -- no codegen, no nimsimd import, until
   something instantiates them (see the amendment above).
2. The algorithm modules (`naivesum`, `pairwisesum`, `compensatedsum`) never
   import `simd` — SIMD dispatch happens at the umbrella and C ABI only.
3. `simdF64Enabled` is false on arm64; the C ABI falls back to scalar there.
4. Compensated SIMD returns `(T, bool)` and the C ABI falls back to the scalar
   algorithm when `reliable = false`.
5. No FMA in the compensated-sum recurrence; the SIMD dot kernels (ADR-0007)
   use FMA intrinsics.
6. `-d:scalarUniAccurate` opts out of the FMA and `-mavx*` flags; do not pair it
   with `-d:simd` (the umbrella still imports `simd.nim` under `-d:simd` alone).
