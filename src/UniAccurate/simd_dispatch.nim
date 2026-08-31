# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Runtime AVX2/AVX-512 dispatch for the dot family (`naiveDot`/`dot2`/
## `dotK(_, _, 3)`), consumed by `c_api.nim` -- ADR-0008.
##
## Independent of `-d:simd`/`-d:avx2`/`-d:avx512`: on amd64 both ISA variants
## are compiled unconditionally into one translation unit via a per-function
## `target` attribute (`simd.nim`'s `defineNaiveDotV`/`defineDot2V`/
## `defineDotK3V`, the same templates the existing `-d:simd` compile-time
## path instantiates), and `nimsimd/runtimecheck`'s real CPUID probe picks
## one on first call, cached in a module-level proc variable. No build flag
## is needed for this path -- `dispatchNaiveDot`/`dispatchDot2`/`dispatchDotK3`
## are the new backing for `ua_dot_naive`/`ua_dot2`/`ua_dot_k` (k = 2, 3)
## unconditionally on amd64, so the C ABI and the Python wheel built from it
## get AVX2/AVX-512 automatically.
##
## Scope: dot family only. The sum family (`naiveSumSimd`/`kahanSumSimd`/
## `neumaierSumSimd`/`kleinSumSimd`) and the pure-Nim `naiveDot`/`dot2`/
## `dotK` are untouched -- calling them directly still requires the existing
## `-d:simd -d:avx2`/`-d:avx512` compile-time-locked opt-in. `dotK` for
## `k` outside `{2, 3}` also stays on the scalar cascade; no generic-K SIMD
## kernel exists (mirrors `simd.nim`'s own `dotK3Simd`, K = 3 only).
##
## Validated on FreeBSD/Zen4 (real AVX-512 hardware): the plain-float64,
## scalar-returning kernel shape (`naiveDot`) measured 2.87x (AVX2) /
## 3.14x (AVX-512) SIMD-only, negligible dispatch overhead. `dot2`/`dotK3`
## measured 5.09x/7.47x and 3.96x/5.05x (SIMD-only / vs. real-world default)
## for that same shape without the `(float64, bool)` reliability tuple this
## module adds. The tuple-returning shape was validated on the same class of
## hardware in a later round -- both ISA variants emit their own registers
## under the per-function `target` attribute alone -- see ADR-0008's
## Verification, including the lane-merge bug that round uncovered.
##
## NEON float64 does not exist (simd.nim's own documented constraint), so
## non-amd64 targets get no acceleration here: `dispatch*` delegates straight
## to the scalar `naiveDot`/`dot2`/`dotK`, identical to today's behavior with
## no `-d:simd` flag at all -- zero change on arm64.
##
## MSVC (`--cc:vcc`) has no equivalent of GCC/Clang's per-function
## `__attribute__((target(...)))`: there is no MSVC mechanism to compile two
## ISA variants of a function into one translation unit and pick between them
## at runtime -- `config.nims`'s own `-d:simd` ISA selection already works
## around this the same way, with a single compile-time `/arch:AVX2`/
## `/arch:AVX512` flag instead of a per-function attribute. So `defined(vcc)`
## falls into the same scalar-only branch as non-amd64: no regression (the C
## ABI never dispatched SIMD on Windows before ADR-0008 either), just no
## AVX2/AVX-512 gain there.
import ./algorithms/dotproduct

when defined(amd64) and not defined(vcc):
  import contracts
  import nimsimd/avx2
  import nimsimd/fma
  import nimsimd/avx512/f
  import nimsimd/runtimecheck
  import ./simd

  defineNaiveDotV(dispatchNaiveDotAvx2, mm256_setzero_pd(), mm256_loadu_pd,
                  mm256_fmadd_pd, mm256_storeu_pd, 4,
                  "__attribute__((target(\"avx2,fma\"))) $# $#$#")
  defineDot2V(dispatchDot2Avx2, mm256_setzero_pd(), mm256_loadu_pd,
              mm256_mul_pd, mm256_fmadd_pd, mm256_add_pd, mm256_sub_pd,
              mm256_storeu_pd, 4,
              "__attribute__((target(\"avx2,fma\"))) $# $#$#")
  defineDotK3V(dispatchDotK3Avx2, mm256_setzero_pd(), mm256_loadu_pd,
               mm256_mul_pd, mm256_fmadd_pd, mm256_add_pd, mm256_sub_pd,
               mm256_storeu_pd, 4,
               "__attribute__((target(\"avx2,fma\"))) $# $#$#")

  defineNaiveDotV(dispatchNaiveDotAvx512, mm512_setzero_pd(), mm512_loadu_pd,
                  mm512_fmadd_pd, mm512_storeu_pd, 8,
                  "__attribute__((target(\"avx512f,fma\"))) $# $#$#")
  defineDot2V(dispatchDot2Avx512, mm512_setzero_pd(), mm512_loadu_pd,
              mm512_mul_pd, mm512_fmadd_pd, mm512_add_pd, mm512_sub_pd,
              mm512_storeu_pd, 8,
              "__attribute__((target(\"avx512f,fma\"))) $# $#$#")
  defineDotK3V(dispatchDotK3Avx512, mm512_setzero_pd(), mm512_loadu_pd,
               mm512_mul_pd, mm512_fmadd_pd, mm512_add_pd, mm512_sub_pd,
               mm512_storeu_pd, 8,
               "__attribute__((target(\"avx512f,fma\"))) $# $#$#")

  type
    NaiveDotFn = proc(x, y: openArray[float64]): float64 {.nimcall, raises: [].}
    DotKFn = proc(x, y: openArray[float64]): SimdResult[float64] {.nimcall,
        raises: [].}

  var naiveDotImpl: NaiveDotFn
  var dot2Impl, dotK3Impl: DotKFn

  proc scalarNaiveDot(x, y: openArray[float64]): float64 {.raises: [].} =
    naiveDot(x, y)

  proc scalarDot2(x, y: openArray[float64]): SimdResult[float64] {.raises: [].} =
    (dot2(x, y), true)

  proc scalarDotK3(x, y: openArray[float64]): SimdResult[float64] {.raises: [].} =
    (dotK(x, y, 3), true)

  proc initDispatch() {.raises: [].} =
    ## Real CPUID (`nimsimd/runtimecheck`), not a compile-time guess: picks
    ## the best ISA the CPU actually running this binary supports. No
    ## separate FMA bit in `InstructionSet` -- `{AVX512F}`/`{AVX2}` stand in
    ## for "ISA + FMA", true on every shipping CPU with those ISAs to date
    ## (same caveat as the validated prototype this replaces).
    ##
    ## Deliberately unsynchronized: `checkInstructionSets` is a pure CPUID
    ## read (the CPU's feature set cannot change mid-process), so every call
    ## computes the identical result. Two threads racing past the `== nil`
    ## check in `dispatchNaiveDot`/`dispatchDot2`/`dispatchDotK3` below may
    ## both run this body, but each write is to one of three independent,
    ## pointer-sized module variables, and every writer publishes the same
    ## value -- never a mix of ISAs across the three. Worst case is a few
    ## redundant CPUID reads on first concurrent use, not an inconsistent or
    ## incorrect dispatch; `std/once` would only add overhead against a race
    ## that is already benign by construction.
    if checkInstructionSets({AVX512F}):
      naiveDotImpl = dispatchNaiveDotAvx512
      dot2Impl = dispatchDot2Avx512
      dotK3Impl = dispatchDotK3Avx512
    elif checkInstructionSets({AVX2}):
      naiveDotImpl = dispatchNaiveDotAvx2
      dot2Impl = dispatchDot2Avx2
      dotK3Impl = dispatchDotK3Avx2
    else:
      naiveDotImpl = scalarNaiveDot
      dot2Impl = scalarDot2
      dotK3Impl = scalarDotK3

  proc dispatchNaiveDot*(x, y: openArray[float64]): float64 {.contractual,
      raises: [].} =
    ## The plain dot product, through the kernel this CPU actually has. The
    ## choice is made once by CPUID at first call and cached in a proc pointer;
    ## nothing here is decided at compile time, so one binary runs on a machine
    ## with AVX-512 and on one without.
    require:
      x.len == y.len
    body:
      if naiveDotImpl == nil: initDispatch()
      naiveDotImpl(x, y)

  proc dispatchDot2*(x, y: openArray[float64]): float64 {.contractual,
      raises: [].} =
    ## The compensated dot product, dispatched the same way. When the chosen
    ## kernel reports its result unreliable, the scalar `dot2` is run instead
    ## and its answer returned -- the compensation is the point of the call,
    ## and a fast wrong number is not a cheaper right one.
    require:
      x.len == y.len
    body:
      if dot2Impl == nil: initDispatch()
      let (r, reliable) = dot2Impl(x, y)
      if reliable: r else: dot2(x, y)

  proc dispatchDotK3*(x, y: openArray[float64]): float64 {.contractual,
      raises: [].} =
    ## The K-fold compensated dot product at K = 3, dispatched and with the
    ## same fallback as `dispatchDot2`.
    require:
      x.len == y.len
    body:
      if dotK3Impl == nil: initDispatch()
      let (r, reliable) = dotK3Impl(x, y)
      if reliable: r else: dotK(x, y, 3)

else:
  proc dispatchNaiveDot*(x, y: openArray[float64]): float64 {.raises: [].} =
    naiveDot(x, y)
  proc dispatchDot2*(x, y: openArray[float64]): float64 {.raises: [].} =
    dot2(x, y)
  proc dispatchDotK3*(x, y: openArray[float64]): float64 {.raises: [].} =
    dotK(x, y, 3)
