# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Error-free transformations (EFT)
##
## The foundation layer of accurate floating-point arithmetic. An EFT computes
## an operation (add, subtract, multiply) and returns both the rounded result
## `s` and the exact rounding error `e`, such that in real arithmetic
##
##     a op b = s + e   (exactly)
##
## with `s = fl(a op b)` (IEEE-754 roundTiesToEven) and `e` exactly
## representable. The summation and dot-product algorithms built on top of this
## module track and compensate for these errors.
##
## Correctness assumptions (IEEE-754)
## ==================================
##
## The EFTs are exact only under all of:
##   1. `roundTiesToEven` (the IEEE-754 default).
##   2. `FLT_EVAL_METHOD == 0` — every operation rounds to its target precision
##      immediately. On x87 (80-bit, `FLT_EVAL_METHOD == 2`) the intermediate
##      `a + b` is computed in extended precision and `e` is not recovered
##      exactly; use SSE2 (default on amd64) or a hardfloat target.
##   3. No `-ffast-math` / reassociation. Compensated arithmetic depends on the
##      exact operation order. Do not add `-ffast-math` via `--passC`; Nim's C
##      backend does not enable it by default. `twoProductFMA` uses the C99
##      `fma`/`fmaf` intrinsics (a single correctly-rounded operation), never a
##      compiler-contracted `a*b + c`, so `-ffp-contract=off` (set in
##      `config.nims` on amd64 and arm64) does not break it.
##   4. Finite inputs. NaN or ±Inf violates the precondition; the result is
##      undefined (debug builds raise `PreConditionDefect`). Overflow of `s`
##      (e.g. `a + b` beyond max normal) yields `s = ±Inf`; the error `e` is
##      then NaN for `twoSum` (`Inf - Inf` appears in the recovery) and `-Inf`
##      for `fastTwoSum` (`b - Inf`). In both cases the identity no longer
##      holds — the postcondition's `fcInf`/`fcNegInf`/`fcNan` short-circuit
##      prevents the `|e| <= ulp(s)` bound from reading `<= ulp(±Inf) = NaN`.
##
## Contracts
## =========
##
## Pre/postconditions use NimContracts (`{.contractual.}`): checked in debug,
## compiled away under `-d:release`/`-d:danger` (zero release overhead). The
## postconditions state checkable consequences of the EFT identity, chiefly
## `|e| <= ulp(s)` (the non-overlap bound). The full real-arithmetic identity
## `s + e == a op b` is verified to the last bit in `tests/test_eft.nim`, not by
## a float-level contract. `ulp`, `split` and `isNonOverlapping` keep
## `runnableExamples` (incompatible with `{.contractual.}`); their correctness
## is checked directly.
##
## References
## ==========
##
## - Dekker, T.J. (1971). "A floating-point technique for extending the
##   available precision". *Numerische Mathematik* 18(3), 224–242.
##   doi:10.1007/BF01397083
## - Møller, O. (1965). "Quasi double-precision in floating point addition".
##   *BIT* 5(1), 37–50. doi:10.1007/BF01975722
## - Knuth, D.E. (1998). *TAOCP Vol. 2: Seminumerical Algorithms*, 3rd ed.,
##   §4.2.2, Theorem C. Addison-Wesley. ISBN 978-0-201-89684-8
## - Ogita, T., Rump, S.M., Oishi, S. (2005). "Accurate sum and dot product".
##   *SIAM J. Sci. Comput.* 26(6), 1955–1988. doi:10.1137/030601818
## - Boldo, S., Melquiond, G. (2008). "Emulation of a FMA and correctly rounded
##   sums: proved algorithms using rounding to odd". *IEEE Trans. Computers*
##   57(4), 462–471. doi:10.1109/TC.2007.70819
## - Shewchuk, J.R. (1997). "Adaptive Precision Floating-Point Arithmetic and
##   Fast Robust Geometric Predicates". *Discrete Comput. Geom.* 18(3),
##   305–363. doi:10.1007/PL00009321
## - Goldberg, D. (1991). "What Every Computer Scientist Should Know About
##   Floating-Point Arithmetic". *ACM Comput. Surv.* 23(1), 5–48.
##   doi:10.1145/103162.103163

import std/math
import contracts

const
  SplitFactor64* = 134217729.0
    ## Veltkamp split factor for float64: `2^27 + 1`. Splits a 53-bit
    ## significand into two non-overlapping parts (26 high, 27 low bits).
    ## Dekker (1971).
  SplitFactor32* = 4097.0'f32
    ## Veltkamp split factor for float32: `2^12 + 1`. Splits a 24-bit
    ## significand into two non-overlapping 12-bit parts. Must be a float32 so
    ## `SplitFactor32 * a` stays in float32 (a float64 literal would promote
    ## `a` and compute the split in float64, silently breaking the 12-bit
    ## split).

func ulp*[T: SomeFloat](x: T): T {.inline.} =
  ## Unit in the Last Place of `x`: the spacing of floats in the binade
  ## containing `|x|` — the "ulp above" convention (Goldberg 1991). Constant
  ## within a binade: `ulp(1.0) = 2^-52`, `ulp(1.5) = 2^-52`, `ulp(2.0) =
  ## 2^-51`. For `x == 0` or subnormal `x`, returns the smallest positive
  ## subnormal. For NaN/Inf returns NaN.
  ##
  ## Bit-exact (no logarithms, no `|x| * 2^-52` approximation): the spacing is
  ## read from the exponent field, correct across the normal range, the
  ## subnormal range, and the binade boundary where the spacing doubles.
  runnableExamples:
    doAssert ulp(1.0) == 2.220446049250313e-16 # 2^-52
    doAssert ulp(2.0) == 4.440892098500626e-16 # 2^-51
    doAssert ulp(0.0) == 5e-324 # smallest float64 subnormal

  if classify(x) in {fcNan, fcInf, fcNegInf}:
    return NaN
  let ax = abs(x)
  when T is float64:
    const MantBits = 52
    let bits = cast[uint64](ax)
    let expField = (bits shr MantBits) and 0x7FF'u64
    var u: uint64
    if expField == 0: # zero or subnormal
      u = 1'u64 # 2^-1074 (min subnormal)
    elif expField >= MantBits + 1: # ulp is a normal power of two
      u = (expField - MantBits) shl MantBits
    else: # ulp is subnormal
      u = 1'u64 shl (expField - 1)
    result = cast[T](u)
  else:
    const MantBits = 23
    let bits = cast[uint32](ax)
    let expField = (bits shr MantBits) and 0xFF'u32
    var u: uint32
    if expField == 0:
      u = 1'u32 # 2^-149 (min subnormal)
    elif expField >= MantBits + 1:
      u = (expField - MantBits) shl MantBits
    else:
      u = 1'u32 shl (expField - 1)
    result = cast[T](u)

func isFin*(x: SomeFloat): bool {.inline.} =
  ## Finite: normal, subnormal, or zero — excludes NaN and ±Inf. The guard the
  ## EFT precondition rests on: a compensated algorithm that meets a non-finite
  ## operand or partial sum diverts that step to a plain IEEE `+`, so NaN/Inf
  ## propagate as a naive sum would and finite inputs never raise.
  runnableExamples:
    doAssert isFin(1.0)
    doAssert isFin(0.0)
    doAssert isFin(5e-324) # smallest subnormal
    doAssert not isFin(NaN)
    doAssert not isFin(Inf)
    doAssert not isFin(-Inf)
  classify(x) notin {fcNan, fcInf, fcNegInf}

func allFin*[T: SomeFloat](x: openArray[T]): bool {.inline.} =
  ## True iff every element of `x` is finite (`isFin`). The finite-input safety
  ## postcondition of the compensated sums: when `allFin(x)` holds the result is
  ## never NaN (overflow may yield ±Inf, but a single-sign one — the per-step
  ## `isFin` guard keeps an `Inf − Inf = NaN` from ever being evaluated).
  runnableExamples:
    doAssert allFin([1.0, 2.0, 3.0])
    doAssert not allFin([1.0, NaN, 2.0])
    doAssert not allFin([1.0, Inf])
    doAssert allFin(newSeq[float64](0))
  for v in x:
    if not isFin(v):
      return false
  true

func split*[T: SomeFloat](a: T): (T, T) {.contractual, inline.} =
  ## Veltkamp's splitting (Dekker 1971): decompose `a` into `(hi, lo)` with
  ## `a = hi + lo` exactly, `hi` and `lo` bit-disjoint, each holding at most
  ## `ceil(p/2)` significand bits (27 for float64, 12 for float32). Used by
  ## `twoProduct` to obtain an error-free product without hardware FMA.
  ##
  ## Limitation: `SplitFactor * a` overflows for `|a| > 2^(emax - splitBits)`
  ## (~`2^996` float64, `2^115` float32) — far beyond any practical magnitude;
  ## there the identity `hi + lo = a` is not recovered. Near the underflow
  ## threshold `SplitFactor * a` can round, degrading `lo` by a subnormal ULP.
  require:
    classify(a) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
  ensure:
    # Exact for normals below the overflow threshold and for small subnormals.
    # Huge normals overflow `SplitFactor*a` to Inf/NaN (recombination fails);
    # subnormals near underflow can round. Tolerate both, mirroring the EFT
    # primitives' overflow handling.
    classify(result[0]) in {fcInf, fcNegInf, fcNan} or
      classify(a) in {fcSubnormal, fcZero, fcNegZero} or
      result[0] + result[1] == a
  body:
    when T is float32:
      let c = SplitFactor32 * a
    else:
      let c = SplitFactor64 * a
    result[0] = c - (c - a) # hi
    result[1] = a - result[0] # lo

func fastTwoSum*[T: SomeFloat](a, b: T): (T, T) {.contractual, inline.} =
  ## Dekker's FastTwoSum: error-free addition under the precondition
  ## `|a| >= |b|`. Returns `(s, e)` with `a + b = s + e` exactly, `s = fl(a+b)`,
  ## `|e| <= ½ ulp(s)` for normal `s` (1 ulp subnormal, where ½ ulp is not
  ## representable).
  ##
  ## Cheaper than `twoSum` (3 FLOPs vs 6) but requires magnitude-ordered
  ## operands; if `|a| < |b|` the result is wrong — use `twoSum`. Dekker (1971).
  require:
    classify(a) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
    classify(b) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
    abs(a) >= abs(b)
  ensure:
    classify(result[0]) in {fcInf, fcNegInf, fcNan} or abs(result[1]) <= ulp(
        result[0])
  body:
    result[0] = a + b
    result[1] = b - (result[0] - a)

func twoSum*[T: SomeFloat](a, b: T): (T, T) {.contractual, inline.} =
  ## Møller–Knuth TwoSum: error-free addition with no precondition on operand
  ## ordering. Returns `(s, e)` with `a + b = s + e` exactly, `s = fl(a+b)`,
  ## `|e| <= ½ ulp(s)` for normal `s` (1 ulp subnormal). 6 FLOPs, branchless.
  ##
  ## Møller (1965); Knuth (1998) TAOCP 2, §4.2.2, Theorem B.
  require:
    classify(a) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
    classify(b) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
  ensure:
    classify(result[0]) in {fcInf, fcNegInf, fcNan} or abs(result[1]) <= ulp(
        result[0])
  body:
    result[0] = a + b
    let z = result[0] - a
    result[1] = (a - (result[0] - z)) + (b - z)

func twoSumFast*[T: SomeFloat](a, b: T): (T, T) {.contractual, inline.} =
  ## Branched FastTwoSum: error-free addition, unconditionally correct (a
  ## magnitude swap restores the FastTwoSum precondition). Returns the same
  ## `(s, e)` as `twoSum` (the values are identical; only the sign of a zero
  ## error may differ between the two formulations), but in 3 FLOPs + 1
  ## magnitude branch instead of 6 branchless FLOPs.
  ##
  ## Throughput, not accuracy: both forms are exact. The branch is
  ## well-predicted when one operand dominates (a running sum vs its addends)
  ## → `twoSumFast` wins; `twoSum` (branchless) wins when `|a|` vs `|b|` flips
  ## often (cancellation data with comparable addends). Dekker (1971).
  require:
    classify(a) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
    classify(b) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
  ensure:
    classify(result[0]) in {fcInf, fcNegInf, fcNan} or abs(result[1]) <= ulp(
        result[0])
  body:
    if abs(a) >= abs(b):
      result = fastTwoSum(a, b)
    else:
      result = fastTwoSum(b, a)

func twoDiff*[T: SomeFloat](a, b: T): (T, T) {.contractual, inline.} =
  ## Møller–Knuth error-free subtraction: `a - b = s + e` exactly, `s =
  ## fl(a-b)`, `|e| <= ½ ulp(s)` for normal `s` (1 ulp subnormal). 6 FLOPs,
  ## branchless — the subtraction analogue of `twoSum`.
  require:
    classify(a) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
    classify(b) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
  ensure:
    classify(result[0]) in {fcInf, fcNegInf, fcNan} or abs(result[1]) <= ulp(
        result[0])
  body:
    result[0] = a - b
    let z = result[0] - a
    result[1] = (a - (result[0] - z)) - (b + z)

func twoProduct*[T: SomeFloat](a, b: T): (T, T) {.contractual, inline.} =
  ## Dekker's error-free multiplication (no FMA): `a * b = p + e` exactly,
  ## `p = fl(a*b)`, `|e| < ulp(p)`. Uses `split` on each operand; 17 FLOPs.
  ##
  ## Exactness limits:
  ##   * Total product underflow — `a*b` below the smallest subnormal, so
  ##     `fl(a*b) = 0` while the exact product is nonzero — cannot carry an
  ##     error; it is rejected by the precondition. Gradual underflow of the
  ##     split-products (`ah*bl`, `al*bh`, `al*bl`) only degrades `e` by a
  ##     subnormal ULP; `p` stays exact.
  ##   * Split overflow — `SplitFactor * a` overflows for `|a|` near `emax`,
  ##     poisoning `e` with NaN. On FMA-capable targets the overflow regime
  ##     delegates to `twoProductFMA` (which needs no split) for the exact
  ##     result; on targets without a cheap FMA it is a documented Dekker
  ##     limitation, and the postcondition does not assert the bound there.
  ## Dekker (1971).
  require:
    classify(a) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
    classify(b) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
    a == T(0) or b == T(0) or a * b != T(0) # reject total product underflow
  ensure:
    classify(result[0]) in {fcInf, fcNegInf, fcNan} or
      classify(result[1]) in {fcNan} or # split-overflow regime (no-FMA fallback)
      abs(result[1]) <= ulp(result[0])
  body:
    result[0] = a * b
    let (ah, al) = split(a)
    let (bh, bl) = split(b)
    when defined(useFMA) or defined(amd64) or defined(arm64):
      # Split overflowed (|operand| near emax): the Dekker error is NaN. Use
      # the FMA path, which needs no split and is exact here.
      if classify(ah) == fcNan or classify(bh) == fcNan:
        return twoProductFMA(a, b)
    result[1] = ((ah * bh - result[0]) + ah * bl + al * bh) + al * bl

# `twoProductFMA` uses the C99 libm `fma`/`fmaf`, which compute `a*b + c` with a
# single rounding (correctly rounded per IEEE-754) whether or not the CPU has a
# hardware FMA unit — without one, libm emulates it (correct, slower). The
# guard selects the FMA path on platforms with a cheap libm FMA (amd64/arm64)
# or an explicit `useFMA` override; other targets fall back to the 17-FLOP
# `twoProduct`. Both are exact. Ogita, Rump, Oishi (2005) §3, Algorithm 3.5;
# Boldo & Melquiond (2008).

when defined(useFMA) or defined(amd64) or defined(arm64):

  {.push checks: off.} # libm intrinsics do not overflow in Nim's sense.

  func libmFma(a, b, c: float64): float64 {.importc: "fma", header: "<math.h>".}
    ## Fused multiply-add: `fl(a*b + c)` with a single rounding (C99/IEEE-754).
  func libmFmaf(a, b, c: float32): float32 {.importc: "fmaf",
      header: "<math.h>".}
    ## Fused multiply-add for float32.

  {.pop.}

  func twoProductFMA*(a, b: float64): (float64, float64) {.contractual, inline.} =
    ## FMA-accelerated error-free multiplication (float64), 2 FLOPs. Exact for
    ## every finite product that does not totally underflow; a product below
    ## the smallest subnormal is rejected (see `twoProduct`'s exactness limits).
    require:
      classify(a) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
      classify(b) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
      a == 0.0 or b == 0.0 or a * b != 0.0
    ensure:
      classify(result[0]) in {fcInf, fcNegInf, fcNan} or abs(result[1]) <= ulp(
          result[0])
    body:
      result[0] = a * b
      result[1] = libmFma(a, b, -result[0])

  func twoProductFMA*(a, b: float32): (float32, float32) {.contractual, inline.} =
    ## FMA-accelerated error-free multiplication (float32), 2 FLOPs. Exact for
    ## every finite product that does not totally underflow; a product below
    ## the smallest subnormal is rejected.
    require:
      classify(a) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
      classify(b) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
      a == 0.0'f32 or b == 0.0'f32 or a * b != 0.0'f32
    ensure:
      classify(result[0]) in {fcInf, fcNegInf, fcNan} or abs(result[1]) <= ulp(
          result[0])
    body:
      result[0] = a * b
      result[1] = libmFmaf(a, b, -result[0])

else:
  # Targets without a cheap libm FMA: alias to the Dekker product.
  func twoProductFMA*[T: SomeFloat](a, b: T): (T, T) {.inline.} =
    twoProduct(a, b)

func isNonOverlapping*[T: SomeFloat](a, b: T): bool {.inline.} =
  ## Bit-disjoint (Shewchuk) non-overlap test: two finite floats `a`, `b` are
  ## non-overlapping iff they have no `1` bit in the same binary position —
  ## the property that makes an error-free expansion a valid exact
  ## representation, and that holds for every EFT output here (`s` and `e`
  ## from `twoSum`/`fastTwoSum` never share a set bit).
  ##
  ## Weaker than the `abs(b) < ulp(abs(a))` "non-adjacent" test: `1.0` and
  ## `0.5` are non-overlapping (bits in different positions) yet not
  ## non-adjacent. NaN/Inf are reported as overlapping (undefined for them); a
  ## finite zero paired with anything is trivially non-overlapping. Shewchuk
  ## (1997).
  runnableExamples:
    let (s, e) = twoSum(1.0, 1e-20)
    doAssert isNonOverlapping(s, e)
    let (hi, lo) = split(1.0)
    doAssert isNonOverlapping(hi, lo)

  if classify(a) in {fcNan, fcInf, fcNegInf} or
      classify(b) in {fcNan, fcInf, fcNegInf}:
    return false
  if a == T(0) or b == T(0):
    return true
  when T is float64:
    type U = uint64
    const MantBits = 52
    const ExpMask: U = 0x7FF
    const Bias = 1023
    const SubExp = -1022 - 52 # subnormal LSB position (2^-1074)
  else:
    type U = uint32
    const MantBits = 23
    const ExpMask: U = 0xFF
    const Bias = 127
    const SubExp = -126 - 23 # subnormal LSB position (2^-149)

  proc decode(x: T): tuple[sig: U, exp: int] =
    ## `x = sig * 2^exp` with `sig` the integer significand (implicit bit
    ## included for normals) and `exp` the bit position of `sig`'s LSB.
    let bits = cast[U](abs(x))
    let expField = int((bits shr MantBits) and ExpMask)
    let frac = bits and ((U(1) shl MantBits) - 1)
    if expField == 0: # subnormal (zero excluded above)
      (frac, SubExp)
    else:
      (frac or (U(1) shl MantBits), expField - Bias - MantBits)

  var (sigA, expA) = decode(a)
  var (sigB, expB) = decode(b)
  if expB > expA:
    swap(sigA, sigB)
    swap(expA, expB)
  let shift = expA - expB
  if shift > MantBits: # smaller entirely below larger's LSB
    return true
  result = (sigA and (sigB shr shift)) == 0
