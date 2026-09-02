# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## C ABI for UniAccurate. Built --app:staticlib/--app:lib --noMain --mm:arc -d:release.
## Keep in sync with include/UniAccurate.h; tests/c links the header against this lib.
import std/math
import ../UniAccurate
import ./simd_dispatch

const UniAccurateVersionC: cstring = "1.1.0"

# Unmangled C symbols, C calling convention, exported from the shared lib.

# --noMain suppresses the generated entry point and with it every auto-init
# hook: neither the static nor the shared build emits a DllMain or an ELF
# constructor, so nothing initializes the Nim runtime. The first entry point
# then enters Nim code whose globals were never set up. The shared build was
# assumed to be covered by a loader hook it does not have -- its registries
# stayed empty and the contrast entry answered nan. Every --noMain task passes
# -d:noAutoInit; an ordinary executable linking this module must not, since its
# own main already ran NimMain.
when defined(noAutoInit):
  # A once primitive, not a plain flag: two threads reaching an entry point
  # together would both see the flag unset, both call NimMain, and the second
  # would enter Nim code the first had not finished initializing. The platform
  # primitives block the losers until the winner returns, which a flag cannot.
  #
  # C statics, not Nim globals: module initialization would reset a Nim one and
  # NimMain would run again. NimMain is declared here too — the generated
  # prototype comes after this section.
  {.emit: """/*VARSECTION*/
void NimMain(void);
#ifdef _WIN32
#  include <windows.h>
static INIT_ONCE ua_runtime_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK ua_runtime_init(PINIT_ONCE o, PVOID p, PVOID *c) {
  (void)o; (void)p; (void)c; NimMain(); return TRUE;
}
static void ua_runtime_ensure(void) {
  InitOnceExecuteOnce(&ua_runtime_once, ua_runtime_init, NULL, NULL);
}
#else
#  include <pthread.h>
static pthread_once_t ua_runtime_once = PTHREAD_ONCE_INIT;
static void ua_runtime_init(void) { NimMain(); }
static void ua_runtime_ensure(void) {
  pthread_once(&ua_runtime_once, ua_runtime_init);
}
#endif
""".}
  template ensureRuntime() =
    {.emit: "  ua_runtime_ensure();".}
else:
  template ensureRuntime() = discard


{.push exportc, cdecl, dynlib.}

proc ua_two_sum(a, b: cdouble; s, e: ptr cdouble) {.raises: [].} =
  ## Error-free sum: writes `*s = fl(a+b)` and `*e` with `a + b = *s + *e`
  ## exactly in real arithmetic (Møller–Knuth TwoSum). For non-finite `a` or
  ## `b`, `*s = a + b` (IEEE) and `*e = NaN`. Never raises. Null `s` or `e` is
  ## undefined.
  ensureRuntime()
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
  ensureRuntime()
  UniAccurateVersionC

proc ua_sum_naive(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Naive (sequential) sum of `n` doubles at `x`. Empty input is `0`. Never
  ## raises; NaN/Inf propagate. Null `x` with `n > 0` is undefined.
  ensureRuntime()
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
  ensureRuntime()
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
  ensureRuntime()
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  pairwiseSumIterative(toOpenArray(arr, 0, last))

proc ua_sum_kahan(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Kahan compensated sum of `n` doubles at `x`. Empty input is `0`. Never
  ## raises; NaN/Inf propagate. Finite inputs never yield NaN (overflow ⇒ ±Inf).
  ## Null `x` with `n > 0` is undefined.
  ensureRuntime()
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
  ensureRuntime()
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
  ensureRuntime()
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
  ## under `-d:simd` (ADR-0006).
  ensureRuntime()
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
  ## with `n > 0` is undefined. No SIMD kernel — scalar under `-d:simd` (ADR-0006).
  ensureRuntime()
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
  ## scalar under `-d:simd` (ADR-0006).
  ensureRuntime()
  if n == 0:
    return 0.0
  let ax = cast[ptr UncheckedArray[cdouble]](x)
  let ay = cast[ptr UncheckedArray[cdouble]](y)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  superDot(toOpenArray(ax, 0, last), toOpenArray(ay, 0, last))

proc ua_dot_naive(x, y: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Naive dot product `Σ xᵢyᵢ` of `n` pairs (left-to-right). Empty input is `0`.
  ## Never raises; NaN/Inf propagate. Finite inputs never yield NaN: a product of
  ## two finite values can overflow to ±Inf, and opposite-sign ±Inf products can
  ## combine to NaN (an order artifact), so on that rare overflow the result is
  ## recovered exactly via `superDot`. Null `x` or `y` with `n > 0` is undefined.
  ## On amd64 an FMA reduce kernel runs on AVX2/AVX-512, picked at runtime via a
  ## real CPUID check (`simd_dispatch.nim`, ADR-0008) -- no `-d:simd` needed; the
  ## `superDot` fallback is shared with the scalar path. Other targets (NEON
  ## float64 has no SIMD path) use the scalar kernel directly.
  ensureRuntime()
  if n == 0:
    return 0.0
  let ax = cast[ptr UncheckedArray[cdouble]](x)
  let ay = cast[ptr UncheckedArray[cdouble]](y)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  template sx: untyped = toOpenArray(ax, 0, last)
  template sy: untyped = toOpenArray(ay, 0, last)
  let r = dispatchNaiveDot(sx, sy)
  if classify(r) == fcNan and allFin(sx) and allFin(sy): superDot(sx, sy) else: r

proc ua_dot2(x, y: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Compensated dot product at twice working precision (ORO Alg. 5.3, K = 2;
  ## Graillat Dot2FMA) of `n` pairs. Empty input is `0`. Never raises; NaN/Inf
  ## propagate. Finite inputs never yield NaN (overflow ⇒ ±Inf, or `superDot` on
  ## opposite-sign product overflow via the `naiveDot` fallback). Null `x` or `y`
  ## with `n > 0` is undefined. On amd64 the Dot2 FMA kernel runs on AVX2/AVX-512,
  ## picked at runtime via a real CPUID check (ADR-0008) -- no `-d:simd` needed --
  ## with a lane-concentration guard that falls back to the scalar `dot2` on
  ## cancellation data. Other targets (NEON float64 has no SIMD path) use the
  ## scalar kernel directly.
  ensureRuntime()
  if n == 0:
    return 0.0
  let ax = cast[ptr UncheckedArray[cdouble]](x)
  let ay = cast[ptr UncheckedArray[cdouble]](y)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  template sx: untyped = toOpenArray(ax, 0, last)
  template sy: untyped = toOpenArray(ay, 0, last)
  dispatchDot2(sx, sy)

proc ua_dot_k(x, y: ptr cdouble; n: csize_t; k: cint): cdouble {.raises: [].} =
  ## K-fold compensated dot product (ORO Alg. 5.3) of `n` pairs: `k = 1` the
  ## naive dot, `k = 2` twice precision (≈ `ua_dot2`), `k = 3` threefold; `k < 1`
  ## is treated as 1. Empty input is `0`. Never raises; NaN/Inf propagate. Finite
  ## inputs never yield NaN. Null `x` or `y` with `n > 0` is undefined. On amd64
  ## `k = 2` dispatches to the Dot2 FMA kernel and `k = 3` to the DotK3 FMA
  ## kernel, each picked at runtime via a real CPUID check (ADR-0008, no
  ## `-d:simd` needed) with a lane-concentration guard that falls back to the
  ## scalar body; other `k` run the scalar cascade on every target (no generic-K
  ## SIMD kernel exists). Other targets (NEON float64 has no SIMD path) always
  ## use the scalar cascade.
  ensureRuntime()
  if n == 0:
    return 0.0
  let ax = cast[ptr UncheckedArray[cdouble]](x)
  let ay = cast[ptr UncheckedArray[cdouble]](y)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  template sx: untyped = toOpenArray(ax, 0, last)
  template sy: untyped = toOpenArray(ay, 0, last)
  if k == 2:
    dispatchDot2(sx, sy)
  elif k == 3:
    dispatchDotK3(sx, sy)
  else:
    dotK(sx, sy, int(k))

proc ua_sum_oro(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## ORO `sum2` (Alg 4.1) — the magnitude-robust compensated sum, value-identical
  ## to `neumaierSum`. Empty input is `0`. Never raises; NaN/Inf propagate.
  ## Finite inputs never yield NaN (overflow ⇒ ±Inf). Null `x` with `n > 0` is
  ## undefined. No SIMD kernel — scalar under `-d:simd` (ADR-0006).
  ensureRuntime()
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  sum2(toOpenArray(arr, 0, last))

proc ua_sum_near(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Rump `NearSum` (Alg 7.4) — the correctly-rounded sum, the round-to-nearest
  ## of the exact `Σ xᵢ` bit-for-bit. Empty input is `0`. Never raises; NaN/Inf
  ## propagate (non-finite input or a `sigma0` overflow falls back to the exact
  ## superaccumulator). Finite inputs never yield NaN (overflow ⇒ ±Inf). Null `x`
  ## with `n > 0` is undefined. No SIMD kernel — scalar under `-d:simd` (ADR-0006).
  ensureRuntime()
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  nearSum(toOpenArray(arr, 0, last))

proc ua_sum_acc(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Rump `AccSum` (Alg 4.5) — a *faithful* rounding of `Σ xᵢ` (within 1 ulp of
  ## the exact sum). Empty input is `0`. Never raises; NaN/Inf propagate
  ## (non-finite input or a `sigma0` overflow falls back to the exact
  ## superaccumulator, which is correctly-rounded and so still faithful). Finite
  ## inputs never yield NaN (overflow ⇒ ±Inf). Null `x` with `n > 0` is undefined.
  ## No SIMD kernel — scalar under `-d:simd` (ADR-0006).
  ensureRuntime()
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  accSum(toOpenArray(arr, 0, last))

proc ua_sum_k(x: ptr cdouble; n: csize_t; k: cint): cdouble {.raises: [].} =
  ## ORO `SumK` (Alg 4.8) — K-fold cascaded compensated sum: K=1 the naive
  ## `twoSum` chain, K=2 first-order compensated (≈ `neumaierSum`), K=3
  ## second-order (≈ `kleinSum`); `k < 1` is treated as 1. Empty input is `0`.
  ## Never raises; NaN/Inf propagate. Finite inputs never yield NaN (overflow ⇒
  ## ±Inf). Null `x` with `n > 0` is undefined. No SIMD kernel — scalar under
  ## `-d:simd` (ADR-0006).
  ensureRuntime()
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  sumK(toOpenArray(arr, 0, last), int(k))

proc ua_condition_number(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ## Condition number of the sum, `cond = Σ|xᵢ| / |Σxᵢ|` (Higham 2002, §4.1): `1`
  ## for a no-cancellation sum, large under catastrophic cancellation, `Inf` when
  ## the sum is exactly `0` or `Σ|xᵢ|` overflows the float range, `0` for empty
  ## input. Never raises; finite inputs never yield NaN. Null `x` with `n > 0` is
  ## undefined.
  ensureRuntime()
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  conditionNumber(toOpenArray(arr, 0, last))

proc ua_norm_scaled(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ensureRuntime()
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  scaledEuclideanNorm(toOpenArray(arr, 0, last))

proc ua_mean_scaled(x: ptr cdouble; n: csize_t): cdouble {.raises: [].} =
  ensureRuntime()
  if n == 0:
    return NaN
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  try:
    scaledMean(toOpenArray(arr, 0, last))
  except ValueError:
    NaN

proc ua_centered_sum_squares(x: ptr cdouble; n: csize_t;
    center: cdouble): cdouble {.raises: [].} =
  ensureRuntime()
  if n == 0:
    return 0.0
  let arr = cast[ptr UncheckedArray[cdouble]](x)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  centeredSumSquares(toOpenArray(arr, 0, last), center)

proc ua_centered_cross_product(x, y: ptr cdouble; n: csize_t; centerX,
    centerY: cdouble): cdouble {.raises: [].} =
  ensureRuntime()
  if n == 0:
    return 0.0
  let ax = cast[ptr UncheckedArray[cdouble]](x)
  let ay = cast[ptr UncheckedArray[cdouble]](y)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  centeredCrossProduct(toOpenArray(ax, 0, last), toOpenArray(ay, 0, last),
    centerX, centerY)

proc ua_centered_cosine_similarity(x, y: ptr cdouble; n: csize_t; centerX,
    centerY: cdouble): cdouble {.raises: [].} =
  ensureRuntime()
  if n == 0:
    return NaN
  let ax = cast[ptr UncheckedArray[cdouble]](x)
  let ay = cast[ptr UncheckedArray[cdouble]](y)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  try:
    centeredCosineSimilarity(toOpenArray(ax, 0, last),
      toOpenArray(ay, 0, last), centerX, centerY)
  except ValueError:
    NaN

proc ua_centered_projection_coefficient(x, y: ptr cdouble; n: csize_t; centerX,
    centerY: cdouble): cdouble {.raises: [].} =
  ensureRuntime()
  if n == 0:
    return NaN
  let ax = cast[ptr UncheckedArray[cdouble]](x)
  let ay = cast[ptr UncheckedArray[cdouble]](y)
  let last = if n > csize_t(high(int)): high(int) - 1 else: int(n) - 1
  try:
    centeredProjectionCoefficient(toOpenArray(ax, 0, last),
      toOpenArray(ay, 0, last), centerX, centerY)
  except ValueError:
    NaN

{.pop.}
