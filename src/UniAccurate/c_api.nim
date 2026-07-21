# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## C ABI for UniAccurate. Built --app:staticlib/--app:lib --noMain --mm:arc -d:release.
## Keep in sync with include/UniAccurate.h; tests/c links the header against this lib.
import std/math
import ../UniAccurate

const UniAccurateVersionC: cstring = "0.1.0"

# Unmangled C symbols, C calling convention, exported from the shared lib.
{.push exportc, cdecl, dynlib.}

proc ua_two_sum(a, b: cdouble; s, e: ptr cdouble) {.raises: [].} =
  ## Error-free sum: writes `*s = fl(a+b)` and `*e` with `a + b = *s + *e`
  ## exactly in real arithmetic (Møller–Knuth TwoSum). For non-finite `a` or
  ## `b`, `*s = a + b` (IEEE) and `*e = NaN`. Never raises. Null `s` or `e` is
  ## undefined.
  if classify(a) in {fcNan, fcInf, fcNegInf} or
      classify(b) in {fcNan, fcInf, fcNegInf}:
    s[] = a + b
    e[] = NaN
  else:
    let (ss, ee) = twoSum(a, b)
    s[] = ss
    e[] = ee

proc ua_version(): cstring {.exportc, cdecl, dynlib.} =
  ## Static version string; do not free.
  UniAccurateVersionC

proc ua_sum_naive(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Naive (sequential) sum of `n` doubles at `x`. Empty input is `0`. Never
  ## raises; NaN/Inf propagate. Null `x` with `n > 0` is undefined.
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  template s: untyped = toOpenArray(arr, 0, last)
  when defined(simd):
    when simdF64Enabled:
      naiveSumSimd(s)
    else:
      naiveSum(s)
  else:
    naiveSum(s)

proc ua_sum_pairwise(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Recursive pairwise sum of `n` doubles at `x`. Empty input is `0`. Never
  ## raises; NaN/Inf propagate (opposite-sign overflow can yield NaN). Null `x`
  ## with `n > 0` is undefined.
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  template s: untyped = toOpenArray(arr, 0, last)
  when defined(simd):
    when simdF64Enabled:
      pairwiseSumSimd(s)
    else:
      pairwiseSum(s)
  else:
    pairwiseSum(s)

proc ua_sum_pairwise_iterative(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Iterative (bottom-up) pairwise sum of `n` doubles at `x`. Empty input is
  ## `0`. Never raises; NaN/Inf propagate. Null `x` with `n > 0` is undefined.
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  pairwiseSumIterative(toOpenArray(arr, 0, last))

proc ua_sum_kahan(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Kahan compensated sum of `n` doubles at `x`. Empty input is `0`. Never
  ## raises; NaN/Inf propagate. Finite inputs never yield NaN (overflow ⇒ ±Inf).
  ## Null `x` with `n > 0` is undefined.
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  template s: untyped = toOpenArray(arr, 0, last)
  when defined(simd):
    when simdF64Enabled:
      let (r, reliable) = kahanSumSimd(s)
      if reliable: r else: kahanSum(s)
    else:
      kahanSum(s)
  else:
    kahanSum(s)

proc ua_sum_neumaier(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Kahan-Babuska-Neumaier (magnitude-robust) compensated sum of `n` doubles at
  ## `x`. Empty input is `0`. Never raises; NaN/Inf propagate. Finite inputs
  ## never yield NaN. Null `x` with `n > 0` is undefined.
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  template s: untyped = toOpenArray(arr, 0, last)
  when defined(simd):
    when simdF64Enabled:
      let (r, reliable) = neumaierSumSimd(s)
      if reliable: r else: neumaierSum(s)
    else:
      neumaierSum(s)
  else:
    neumaierSum(s)

proc ua_sum_klein(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Klein two-level compensated sum of `n` doubles at `x`. Empty input is `0`.
  ## Never raises; NaN/Inf propagate. Finite inputs never yield NaN. Null `x`
  ## with `n > 0` is undefined.
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  template s: untyped = toOpenArray(arr, 0, last)
  when defined(simd):
    when simdF64Enabled:
      let (r, reliable) = kleinSumSimd(s)
      if reliable: r else: kleinSum(s)
    else:
      kleinSum(s)
  else:
    kleinSum(s)

proc ua_sum_shewchuk(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Correctly-rounded sum of `n` doubles at `x` via Shewchuk's adaptive-precision
  ## expansion (the CPython `fsum` recipe): `Σ xᵢ` rounded once under
  ## round-to-nearest-even, for finite non-overflowing input. Empty input is
  ## `0`. Never raises; NaN/Inf propagate. An intermediate overflow abandons the
  ## exact expansion and yields a correctly-signed ±Inf (finite input never
  ## yields NaN). Null `x` with `n > 0` is undefined. No SIMD kernel — scalar
  ## under `-d:simd` (ADR-0007).
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  shewchukSum(toOpenArray(arr, 0, last))

proc ua_sum_exact(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Correctly-rounded sum of `n` doubles at `x` via the Neal small
  ## superaccumulator (integer exact accumulation, single final rounding). Empty
  ## input is `0`. Never raises; NaN/Inf propagate. Finite inputs never yield NaN
  ## (true overflow ⇒ ±Inf, opposite-sign overflow cancels exactly). Null `x`
  ## with `n > 0` is undefined. No SIMD kernel — scalar under `-d:simd` (ADR-0007).
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  superSum(toOpenArray(arr, 0, last))

proc ua_dot_exact(x, y: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Correctly-rounded dot product `Σ xᵢyᵢ` of `n` pairs via the Neal small
  ## superaccumulator (64×64→128 product accumulation, single final rounding).
  ## Empty input is `0`. Never raises; NaN/Inf operands propagate. Finite operands
  ## never yield NaN (products held at true magnitude, opposite-sign overflow
  ## cancels exactly). Null `x` or `y` with `n > 0` is undefined. No SIMD kernel —
  ## scalar under `-d:simd` (ADR-0007).
  if n == 0:
    return 0.0
  let ax = cast[ptr UncheckedArray[cdouble]](x)
  let ay = cast[ptr UncheckedArray[cdouble]](y)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  superDot(toOpenArray(ax, 0, last), toOpenArray(ay, 0, last))

{.pop.}
