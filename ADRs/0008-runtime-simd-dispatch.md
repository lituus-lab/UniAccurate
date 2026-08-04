<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0008: Runtime AVX2/AVX-512 dispatch for the C ABI dot family

- Status: Accepted (hardware-validated on amd64/Zen4, dot2/dotK3 reliability
  tuple included -- see Verification below)
- Date: 2026-08-04
- Scope: `src/UniAccurate/simd.nim` (templates hoisted above the `-d:simd`
  gate), `src/UniAccurate/simd_dispatch.nim` (new), `src/UniAccurate/c_api.nim`
  (`ua_dot_naive`/`ua_dot2`/`ua_dot_k`)

## Decision

`ua_dot_naive`, `ua_dot2`, and `ua_dot_k` (for `k = 2` or `k = 3`) now pick
between an AVX2 kernel, an AVX-512 kernel, and the scalar body **at runtime**,
via a real CPUID probe (`nimsimd/runtimecheck.checkInstructionSets`), on amd64
-- with **no build flag required**. This replaces those three functions' prior
`-d:simd`/`-d:avx2`/`-d:avx512` compile-time-locked path, so a plain `pip
install uniaccurate` wheel (built with no special flags) gets AVX2/AVX-512
automatically on hardware that has it, and falls back cleanly on hardware that
doesn't.

### Why runtime dispatch instead of `-d:simd`

The existing `-d:simd -d:avx2`/`-d:avx512` compile-time path (ADR-0005) bakes
one ISA into the binary at build time. That's fine for a Nim consumer who
controls their own build, but the C ABI feeds `pip install uniaccurate`: the
published wheel is one binary distributed to unknown hardware, and it has
never passed `-d:simd` at all -- `clib`/`pyLib`/`buildCython`/`pyWheel` in
`UniAccurate.nimble` pass no such flag. Every wheel user has been getting the
pure scalar path regardless of their CPU.

### How

Per-function GCC/Clang `target` attributes
(`__attribute__((target("avx2,fma")))` / `__attribute__((target("avx512f,fma")))`,
injected via Nim's `codegenDecl` pragma) let both ISA variants of a kernel
compile into one translation unit without a global `-mavx2`/`-mavx512f`. On
first call, `checkInstructionSets` (real `cpuid`/`xgetbv`, not a compile-time
guess) picks AVX-512, else AVX2, else scalar, and caches the choice in a
module-level proc-pointer variable -- one branch on every call after that,
measured negligible (~0.4% over a direct call, see Verification).

`simd_dispatch.nim`'s AVX2/AVX-512 kernels are not a second implementation of
the EFT recurrence: they are new instantiations of `simd.nim`'s existing
`defineNaiveDotV`/`defineDot2V`/`defineDotK3V` templates (the same ones
`-d:simd -d:avx2`/`-d:avx512` instantiate), now hoisted above the `-d:simd`
gate with an added optional `targetAttr` template parameter (default
`"$# $#$#"`, `codegenDecl`'s own no-op default -- the existing `-d:simd`
instantiations don't pass it and are byte-for-byte unaffected). One recurrence,
two call sites, matching this repo's existing "no duplicate implementation"
convention (see the buffer-protocol dispatch pattern in `py/uniaccurate/_core.pyx`).

### Scope: dot family, C ABI only

- **Sum family unaffected.** `naiveSumSimd`/`kahanSumSimd`/`neumaierSumSimd`/
  `kleinSumSimd`/`pairwiseSumSimd` still require `-d:simd -d:avx2`/`-d:avx512`
  exactly as before (ADR-0005) -- extending runtime dispatch to them is
  future work, not covered here.
- **Pure-Nim API unaffected.** Calling `naiveDot`/`dot2`/`dotK` directly from
  Nim still means scalar unless the caller opts into `-d:simd -d:avx2`/
  `-d:avx512` and calls `naiveDotSimd`/`dot2Simd`/`dotK3Simd` explicitly. This
  ADR changes `c_api.nim`'s own internal implementation choice only.
- **`dotK` for `k` outside `{2, 3}`** stays on the scalar cascade on every
  target -- no generic-K SIMD kernel exists (mirrors `dotK3Simd`'s own K = 3
  limit).
- **arm64 gets no acceleration here.** `simd_dispatch.nim` only instantiates
  AVX2/AVX-512 kernels under `when defined(amd64) and not defined(vcc)`; its
  `else:` branch (arm64, and MSVC on amd64) delegates straight to the scalar
  `naiveDot`/`dot2`/`dotK`, identical to today's behavior with no `-d:simd`
  flag -- zero change on arm64, confirmed by the existing `tests/c` suite
  passing unchanged (see Verification). This ADR simply doesn't select an
  arm64 runtime kernel; extending runtime dispatch there (mirroring `simd.nim`'s
  existing float32-only NEON dot kernels) is future work, not covered here.

### Reliability and the NaN-recovery fallback

`dispatchDot2`/`dispatchDotK3` carry the same `(result, reliable)` contract as
`dot2Simd`/`dotK3Simd` (ADR-0005): `reliable = isFin(r) and maxLane <=
LaneConcentrationFallback * abs(r)`, falling back to the scalar body on
cancellation data or a non-finite intermediate. `dispatchNaiveDot` carries the
same `superDot` NaN-recovery fallback `ua_dot_naive` already applied around
`naiveDotSimd` (an FMA reduce can produce NaN from reordered opposite-sign
`Inf` partials that plain left-to-right scalar summation would not hit in the
same order) -- moved to wrap `dispatchNaiveDot` instead, unchanged logic.

## Verification

**Hardware-validated** (FreeBSD/Zen4, real AVX-512): the plain-float64,
scalar-returning kernel shape (`naiveDot`) -- correct (diffs ~1e-12, expected
reordering), dispatches to AVX-512 via real CPUID, negligible dispatch
overhead. Measured: scalar 0.5864 ns/elem, AVX2 0.2042 (2.87x), AVX-512 0.1869
(3.14x). `dot2`/`dotK3` measured, for that same scalar-returning shape (no
`(float64, bool)` tuple, no lane-concentration check): 5.09x/7.47x and
3.96x/5.05x (SIMD-only / vs. real-world default), cross-validated three ways
against ADR-0007's own independent FMA-lever measurement.

**The reliability-tuple round (2026-08-04)**, on FreeBSD 16.0-CURRENT amd64,
AMD Ryzen 9 7900X (Zen4, `avx2 avx512f avx512dq avx512vl fma`), Nim 2.3.1 /
clang. Applying `codegenDecl`'s `target` attribute to a `(float64, bool)`-
returning func -- the shape this ADR ships for `dispatchDot2`/`dispatchDotK3`,
and the one the earlier round did not cover -- was exercised for real:

- **Codegen.** `objdump -d libUniAccurate.a` on the default `nimble clibStatic`
  build (no `-d:simd`, no `-m` flag): `dispatchDot2Avx512` 46 `zmm` operands,
  `dispatchDotK3Avx512` 71, `dispatchNaiveDotAvx512` 4; the AVX2 variants
  `ymm`-only (41 / 66 / 3) with zero `zmm`. Both ISA variants of the
  tuple-returning kernels coexist in one translation unit, per-function
  `target` attribute only, as designed.
- **Selection.** `checkInstructionSets({AVX512F})` returns true on this box, so
  the AVX-512 kernels (L = 8) are the ones actually running.
- **Correctness.** `nimble testAll` 598 checks / 0 failures (Nim debug +
  release + the C ABI suite), `nimble testSimd` 284, `nimble ctest` green, and
  the exact-rational Python oracle 899/899 through the C ABI.

**A real bug surfaced in this round, in the lane merge, and is fixed here.**
`defineDot2V`/`defineDotK3V` collapsed each lane to `sl[j] + el[j]` (resp.
`sl[j] + esl[j] + ecl[j]`) *before* the `neumaierSum` merge, rounding away the
compensation the kernel had just computed. The cost is ~`eps * max|sl|`, which
is invisible on well-conditioned data but not under cancellation, where
`max|sl| >> |result|`. The exact-rational oracle caught it as 9 forward-error-
bound violations (`dot_cancel_{10,100,500}` x `dot2`/`dot_k` K = 2, 3), all on
the amd64 SIMD path only. On `dot_cancel_500` (n = 500, E = 7.425e+02,
cond = E/|S| = 4.6e+02): dispatched `dot2` err `1.443290e-14` against a
`3.570200e-16` bound (~40x over), while the scalar `dot2`/`dotK3` were exact
(err = 0) on the same input. The lane-concentration guard did not fire --
`maxLane/|r| = 61.06`, well under `LaneConcentrationFallback = 1024` -- so the
result was returned as `reliable`, which is exactly why a threshold heuristic
could not substitute for merging correctly.

The fix merges the `2L` (dot2) / `3L` (dotK3) lane parts as separate addends
instead of pre-collapsing them, restoring err = 0 on that case and clearing all
9 oracle failures. It is a correction to the ADR-0005 kernels themselves, so it
applies to the `-d:simd` path too, not just this ADR's runtime dispatch.

`tests/test_simd_dispatch.nim` gained a "past the vector width" suite: the
pre-existing cases topped out at n = 4, which never reaches the strided loop of
an 8-lane AVX-512 kernel, so no Nim-tier test could have caught this. The new
cases run integer-exact agreement for every n in 0..40 and near-cancelling data
at n = 8/16/64/500, on both sides of both vector widths (L = 4, L = 8).

Pre-existing failures on this box, confirmed present at `b3ee77f` (before this
ADR) and therefore out of its scope: `nimble simdAvx2`/`simdAvx512` fail
"naive and compensated, signed big float64 up to n=10000" (NaN, SIMD *sum*
family), `nimble ctestSimd` fails to compile (`-d:simd` C ABI, two structurally
identical `(float64, bool)` tuples get distinct mangled C struct names), and
`nimble lint` reports `simd.nim`/`simd_dispatch.nim` as nimpretty-dirty.

## Consequences

- `pip install uniaccurate` gets AVX2/AVX-512 dot products automatically, no
  build flag, no wheel-matrix change.
- `simd.nim` is no longer an empty module without `-d:simd` (see ADR-0005's
  amendment) -- the dot templates and their small import set are now
  unconditional, at zero cost until instantiated.
- The superseded prototype (`src/UniAccurate/simd_runtime_experiment.nim` and
  its `simdRuntimeExperiment` nimble task) is removed; its validated findings
  are folded into this ADR and into `simd_dispatch.nim`'s doc comment.
