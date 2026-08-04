<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0008: Runtime AVX2/AVX-512 dispatch for the C ABI dot family

- Status: Accepted (naiveDot path hardware-validated; dot2/dotK3 reliability
  shape not yet hardware-validated -- see Verification below)
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

**Not yet hardware-validated**: applying `codegenDecl`'s `target` attribute to
a `(float64, bool)`-returning func (the reliability-tuple shape this ADR
actually ships for `dispatchDot2`/`dispatchDotK3`) is a new combination beyond
what ran on the FreeBSD/Zen4 box -- only the scalar-returning shape was tested
there. What has been verified without amd64 hardware:

- `nim check --cpu:amd64 --os:linux` (full semantic check, no codegen) passes
  for `simd.nim`, `simd_dispatch.nim`, and `c_api.nim`, in every relevant
  build config (`-d:release` alone; `-d:release -d:simd -d:avx2`; `-d:release
  -d:simd -d:avx512`).
- On this arm64 development machine: `simd_dispatch.nim`'s `else:` (non-amd64)
  branch, which shares the same public API and fallback logic as the amd64
  branch, compiles and runs correctly, and the full `nimble testAll` +
  `nimble ctest` + `nimble pyTest` suites (899 Python tests, the C ABI dot/
  dot2/dotK cancellation and overflow edge cases) pass unchanged end to end.
  This proves the fallback path and the surrounding `c_api.nim` wiring, not
  the AVX2/AVX-512 codegen itself.

A fresh FreeBSD/Zen4 run exercising `ua_dot2`/`ua_dot_k(k=2)`/`ua_dot_k(k=3)`
through the real C ABI (not the old scalar-returning prototype shape) is the
remaining step before this ADR's Status can drop the caveat.

## Consequences

- `pip install uniaccurate` gets AVX2/AVX-512 dot products automatically, no
  build flag, no wheel-matrix change.
- `simd.nim` is no longer an empty module without `-d:simd` (see ADR-0005's
  amendment) -- the dot templates and their small import set are now
  unconditional, at zero cost until instantiated.
- The superseded prototype (`src/UniAccurate/simd_runtime_experiment.nim` and
  its `simdRuntimeExperiment` nimble task) is removed; its validated findings
  are folded into this ADR and into `simd_dispatch.nim`'s doc comment.
