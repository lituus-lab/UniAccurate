# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Ogita–Rump–Oishi and Rump summation.
##
## Two accuracy tiers over `compensatedsum`, both built on the error-free
## `transform` (ORO Vecsum, Alg 4.6) rather than per-step feedback:
##
##   * `sum2` — ORO Alg 4.1, the magnitude-robust compensated sum. Identical in
##     value to `neumaierSum` (the ORO Vecsum-with-final-correction *is*
##     Neumaier's scheme); provided under the ORO name for API symmetry with
##     `sumK` and the Rump family below.
##   * `sumK` — ORO Alg 4.8, the K-fold cascaded compensated sum. Applies the
##     `transform` K times and sums the partial results, so each extra cascade
##     level compensates one order of rounding error: K=1 is the naive `twoSum`
##     chain, K=2 the first-order compensated sum (≈ `neumaierSum`), K=3 the
##     second-order (≈ `kleinSum`).
##
## Error bound
## ===========
##
## Let `S = Σ xᵢ`, `E = Σ|xᵢ|`, `u` the unit roundoff, and `γ_k = k·u/(1-k·u)`
## (Higham). Under the IEEE-754 assumptions of `twosum` (roundTiesToEven,
## `FLT_EVAL_METHOD == 0`, no `-ffast-math`, finite non-overflowing
## intermediates):
##
##     sum2:    |fl(sum) - S| <= (2u + O(nu²)) · E       (≈ neumaierSum)
##     sumK:    |fl(sumK) - S| <= γ_{n-1}^K · E / (1 - γ_{n-1}^K)
##
## so `sumK` drives the error to `(n·u)^K · E` — polynomially smaller at each
## order, at `K` transforms of `(n-1)` `twoSum` calls. The bound is in `E`, not
## `|S|`: cancellation grows the relative error, not the absolute. For
## correctly-rounded (error in `|S|`, not `E`) use `shewchukSum` or `superSum`.
##
## Correctness assumptions
## ========================
##
## As for `twosum`: `roundTiesToEven`, `FLT_EVAL_METHOD == 0`, no `-ffast-math`.
## These functions accept arbitrary arrays (including NaN/±Inf): a non-finite
## element or a partial sum that overflows degrades that step to a naive IEEE
## `+`, so NaN/Inf propagate as a plain sum would. Finite inputs never raise;
## the result may be ±Inf only on genuine overflow, never NaN — the per-step
## `isFin` guard and the final `isFin(result)` compensation guard keep an
## `Inf − Inf = NaN` from ever being evaluated (an overflow poisons the error
## vector, but the poisoned partial is never added to a non-finite running sum).
##
## Contracts
## =========
##
## `{.contractual.}` (NimContracts, debug-only, compiled away under
## `-d:release`/`-d:danger`): the trivial empty-input case and the safety
## property — under all-finite input the result is never NaN. The accuracy
## bounds above are real-arithmetic and verified against an exact oracle in the
## test suite, not by a float-level contract.
##
## References
## ==========
##
## - Ogita, T., Rump, S.M., Oishi, S. (2005). "Accurate Sum and Dot Product".
##   *SIAM J. Sci. Comput.* 26(6), 1950–1988. doi:10.1137/S0036142903448029
## - Rump, S.M. (2008). "Accurate summation". *Numer. Math.* 110, 385–404.
##   doi:10.1007/s00211-008-0144-z
## - Higham, N.J. (2002). *Accuracy and Stability of Numerical Algorithms*,
##   2nd ed., §4.3. SIAM. ISBN 978-0-89871-521-7
import std/math
import contracts
import ../twosum
import compensatedsum
import naivesum
import exactsum

func transform[T: SomeFloat](x: openArray[T], err: var seq[T],
    useFastTwoSum: static bool = false,
    assumeFinite: static bool = false): T =
  ## ORO Vecsum (Alg 4.6): error-free transform of a sum. Chains `twoSum`
  ## across `x` so that `result + sum(err) == sum(x)` exactly in real arithmetic,
  ## with `result = fl(Σ xᵢ)` the naive sum and `err` (written through the out-
  ## param) the `n-1` rounding errors. A non-finite operand degrades that step to
  ## naive IEEE propagation (the EFT is lost, the error slot is set to 0); a step
  ## whose sum overflows produces a NaN error, which the cascade's `isFin` guard
  ## localizes.
  ##
  ## `useFastTwoSum` swaps the per-step `twoSum` (6 branchless FLOPs) for the
  ## branched `twoSumFast` (3 FLOPs + 1 magnitude branch). Both yield the same
  ## `(s, e)` EFT, so the transform is bit-identical either way — a throughput
  ## lever, not an accuracy one. The branch is well-predicted when the running
  ## `result` dominates its addends (the common cascade shape), and a loss where
  ## `|result|` vs `|x[i]|` flips often (cancellation data with comparable
  ## addends); the flag lets `sumK` pick per workload.
  ##
  ## `assumeFinite` drops the per-step `isFin` guard (the naive-propagation
  ## divert). Bit-identical to the guarded path on finite non-overflowing input
  ## (the guard never fires there); an overflow then seeds `err` with a NaN from
  ## the EFT recovery instead of a 0 — the caller contracts no overflow. See
  ## `kahanSum` for the contract shape.
  let n = x.len
  if n == 0:
    err.setLen(0)
    return T(0)
  result = x[0]
  err.setLen(n - 1) # reuse the caller's buffer; sumK swaps two buffers whose
                    # lengths strictly decrease, so this never reallocates in the
                    # cascade steady state (cap is already large enough).
  for i in 1 ..< n:
    when not assumeFinite:
      if not isFin(result) or not isFin(x[i]):
        result = result + x[i]
        err[i - 1] = T(0)
        continue
    let (s, e) = when useFastTwoSum: twoSumFast(result, x[i])
                 else: twoSum(result, x[i])
    result = s
    err[i - 1] = e

func sum2*[T: SomeFloat](x: openArray[T],
    assumeFinite: static bool = false): T {.contractual.} =
  ## ORO `sum2` (Alg 4.1) — the magnitude-robust compensated sum. Identical in
  ## value to `neumaierSum`: the ORO Vecsum-with-final-correction is Neumaier's
  ## scheme. Exposed under the ORO name for API symmetry with `sumK`.
  ##
  ## Forward bound: `|fl(sum2) - Σxᵢ| <= (2u + O(nu²)) · Σ|xᵢ|`. See `neumaierSum`
  ## for the magnitude-robustness discussion (recovers a small addend dominated
  ## by the running sum where `kahanSum` loses it).
  ##
  ## `assumeFinite` threads straight into `neumaierSum` (this is its ORO alias):
  ## the per-element and final `isFin` guards are dropped on the opt-in path.
  ## Bit-identical to the default on finite non-overflowing input. See
  ## `neumaierSum` / `kahanSum` for the contract shape.
  require:
    not assumeFinite or allFin(x) # opt-in ⇒ finite input
  ensure:
    x.len != 0 or result == T(0)
    assumeFinite or (not allFin(x) or classify(result) !=
        fcNan) # finite ⇒ no NaN (guard path only)
  body:
    result = neumaierSum(x, assumeFinite)

func sumK*[T: SomeFloat](x: openArray[T], K: int,
    useFastTwoSum: static bool = false,
    assumeFinite: static bool = false): T {.contractual.} =
  ## ORO `SumK` (Alg 4.8): K-fold cascaded compensated summation. Applies the
  ## error-free `transform` K times, summing the partial results: K=1 is the
  ## naive `twoSum` chain (≈ `naiveSum`), K=2 the first-order compensated sum
  ## (≈ `neumaierSum`), K=3 the second-order (≈ `kleinSum`), and each further
  ## transform compensates one order of rounding error.
  ##
  ## Forward bound (ORO Thm 4.9): `|fl(sumK) - Σxᵢ| <= γ_{n-1}^K · Σ|xᵢ| /
  ## (1 - γ_{n-1}^K)`, so the error is `(n·u)^K · Σ|xᵢ|`. `K` is the cascade
  ## depth (K >= 1; K < 1 is treated as 1).
  ##
  ## `useFastTwoSum` threads into every `transform` pass: the branched Dekker
  ## `twoSumFast` (3 FLOPs + 1 branch) replaces the branchless `twoSum` (6
  ## FLOPs). Bit-identical to the default — both are the addition EFT — so the
  ## bound above holds unchanged; only the throughput shape differs (see
  ## `transform`). Default off: the branchless form wins on cancellation data.
  ##
  ## `assumeFinite` threads into `transform` (dropping its per-step `isFin`
  ## guard) and drops the cascade's `isFin(result)` compensation guard. Bit-
  ## identical to the default on finite non-overflowing input (neither guard
  ## fires there); an overflow then corrupts the cascade with a NaN error
  ## instead of localizing it. See `kahanSum` for the contract shape; the
  ## default (`false`) keeps the guards and is safe.
  require:
    not assumeFinite or allFin(x) # opt-in ⇒ finite input
  ensure:
    x.len != 0 or result == T(0)
    assumeFinite or (not allFin(x) or classify(result) !=
        fcNan) # finite ⇒ no NaN (guard path only)
  body:
    if x.len == 0:
      return T(0)
    if x.len == 1:
      return x[0]
    let k = if K < 1: 1 else: K
    var cur: seq[T]
    result = transform(x, cur, useFastTwoSum, assumeFinite)
    var nxt: seq[T]
    for _ in 1 ..< k:
      if cur.len == 0:
        break
      let res = transform(cur, nxt, useFastTwoSum, assumeFinite)
      # An overflow poisons the error vector (a `twoSum` whose sum overflows
      # yields a NaN error), so a later partial `res` can be NaN/Inf. Adding it
      # to a running ±Inf would evaluate Inf - Inf = NaN; the true finite sum
      # is unrecoverable past the float range anyway, so skip the compensation
      # once the running sum overflowed (±Inf is the correctly-signed round).
      # The opt-in path drops this guard (the caller promises no overflow).
      when assumeFinite:
        result += res
      else:
        if isFin(result):
          result += res
      swap(cur, nxt) # reuse the two buffers, no per-iteration allocation

# ---------------------------------------------------------------------------
# Rump faithful / correctly-rounded family (Rump–Ogita–Oishi 2008 Part I/II).
# ---------------------------------------------------------------------------

func pow2i[T: SomeFloat](e: int): T =
  ## `2^e` as a float by exponent construction (libm-free). `e` must lie in the
  ## normal exponent range `[1 - bias, bias - 1]`; callers keep it there.
  when T is float64:
    cast[float64](uint64(e + 1023) shl 52)
  else:
    cast[float32](uint32(e + 127) shl 23)

func ceilLog2(k: int): int =
  ## Smallest `m` with `2^m >= k`, for `k >= 1` (libm-free).
  result = 0
  var p = 1
  while p < k:
    p *= 2
    inc result

func nextPowerTwo*[T: SomeFloat](p: T): T =
  ## Rump `NextPowerTwo` (Part I, Alg 3.5): the smallest power of two `>= |p|`
  ## (`2^ceil(log2|p|)`), computed libm-free via the IEEE bit layout. `0` for
  ## `p = 0`; the smallest normal for a subnormal `p`; `Inf` when the next
  ## power of two overflows the exponent range.
  if p == T(0):
    return T(0)
  let ap = abs(p)
  when T is float64:
    const expMask = 0x7FF'u64
    const mantMask = 0x000F_FFFF_FFFF_FFFF'u64
    const minNormal = 0x0010_0000_0000_0000'u64 # 2^-1022
    let b = cast[uint64](ap)
    let exp = int(b shr 52) and int(expMask)
    if exp == 0:
      return cast[float64](minNormal) # subnormal: next power of two is 2^-1022
    if (b and mantMask) == 0:
      return ap # exact power of two
    if exp + 1 >= int(expMask):
      return T(Inf) # 2^(exp+1-1023) overflows the exponent
    return cast[float64](uint64(exp + 1) shl 52)
  else:
    const expMask = 0xFF'u32
    const mantMask = 0x007F_FFFF'u32
    const minNormal = 0x0080_0000'u32 # 2^-126
    let b = cast[uint32](ap)
    let exp = int(b shr 23) and int(expMask)
    if exp == 0:
      return cast[float32](minNormal)
    if (b and mantMask) == 0:
      return ap
    if exp + 1 >= int(expMask):
      return T(Inf)
    return cast[float32](uint32(exp + 1) shl 23)

func nextFloat[T: SomeFloat](x: T): T =
  ## The next representable float strictly greater than `x` (toward +Inf). For
  ## `x >= 0` the bit pattern increases; for `x < 0` it decreases (negative
  ## magnitudes grow as the bits grow, so toward +Inf is the other way). `-0`
  ## maps to the smallest positive subnormal. Assumes finite `x`.
  when T is float64:
    let b = cast[uint64](x)
    if b == 0x8000_0000_0000_0000'u64: # -0.0 -> smallest positive subnormal
      return cast[float64](1'u64)
    if b shr 63 == 1: # negative: toward +Inf is a bit decrement
      return cast[float64](b - 1'u64)
    return cast[float64](b + 1'u64) # non-negative (incl. +0): toward +Inf is a bit increment
  else:
    let b = cast[uint32](x)
    if b == 0x8000_0000'u32:
      return cast[float32](1'u32)
    if b shr 31 == 1:
      return cast[float32](b - 1'u32)
    return cast[float32](b + 1'u32)

func prevFloat[T: SomeFloat](x: T): T =
  ## The previous representable float strictly less than `x` (toward -Inf). For
  ## `x > 0` the bit pattern decreases; for `x <= 0` it increases (the negative
  ## direction). Both zeros map to the smallest negative subnormal (a bit
  ## decrement of `+0` would cross into the NaN bit patterns). Assumes finite `x`.
  when T is float64:
    let b = cast[uint64](x)
    if b == 0'u64: # +0.0 -> smallest negative subnormal
      return cast[float64](0x8000_0000_0000_0001'u64)
    if b shr 63 == 1: # negative (incl. -0): toward -Inf is a bit increment
      return cast[float64](b + 1'u64)
    return cast[float64](b - 1'u64) # positive: toward -Inf is a bit decrement
  else:
    let b = cast[uint32](x)
    if b == 0'u32:
      return cast[float32](0x8000_0001'u32)
    if b shr 31 == 1:
      return cast[float32](b + 1'u32)
    return cast[float32](b - 1'u32)

func extractVector[T: SomeFloat](sigma: T, p: var seq[T]): T =
  ## Rump `ExtractVector` (Part I, Alg 3.4), in place. With `sigma` a power of
  ## two `>= 2^M·max|p_i|`, splits each `p_i` into a high part `q_i` (absorbed
  ## into the returned `tau`) and a residual written back into `p_i`, so
  ## `p_i = q_i + p'_i` exactly and `sum(p) = tau + sum(p)` after the call.
  result = T(0)
  for i in 0 ..< p.len:
    let qi = (sigma + p[i]) - sigma
    p[i] = p[i] - qi
    result = result + qi

func rumpTransform[T: SomeFloat](p: var seq[T],
    offset: T): tuple[tau1: T, tau2: T] =
  ## Rump `Transform` (Part I, Alg 3.3/4.4) with `offset`, in place. The
  ## error-free `ExtractVector` peels the leading bits of `sum(p) + offset` into
  ## the accumulator `t` (seeded with `offset`, so the offset is never itself
  ## extracted — Lemma 7.3's precondition `offset ∈ eps·sigma0·Z`); `sigma`
  ## shrinks by `2^M·eps` each pass until `|t|` carries the sum down to the
  ## residual scale or `sigma` reaches the underflow floor. A final
  ## `FastTwoSum(t^{m-1}, tau^m)` — exact at the stop, where `|t^{m-1}| >=
  ## |tau^m|` (Rump Lemma 3.4) — recovers the last fold's rounding. `p` is
  ## overwritten with the residual `p'`; the result is `(tau1, tau2)` with
  ## `sum(p) + offset = tau1 + tau2 + sum(p')` exactly. Returns `(Inf, 0)` when
  ## the initial extraction unit `sigma0` overflows (the caller falls back).
  let n = p.len
  if n == 0:
    return (offset, T(0))
  var mu = T(0)
  for v in p:
    let a = abs(v)
    if a > mu:
      mu = a
  if mu == T(0):
    return (offset, T(0)) # all-zero vector: the sum is just the offset
  when T is float64:
    const eps = cast[float64](0x3CA0_0000_0000_0000'u64) # 2^-53 (unit roundoff)
    const eta = cast[float64](1'u64) # 2^-1074 (smallest subnormal)
  else:
    const eps = cast[float32](0x3380_0000'u32) # 2^-24 (unit roundoff)
    const eta = cast[float32](1'u32) # 2^-149
  let m = ceilLog2(n + 2) # M = ceil(log2(n+2))
  let sigma0 = nextPowerTwo(mu) * pow2i[T](m)             # 2^(M + ceil(log2 mu))
  if not isFin(sigma0):
    return (T(Inf), T(0)) # sigma0 overflow: the sum overflows; caller falls back
  let sigmaFloor = T(0.5) * (T(1) / eps) * eta # 0.5·eps^-1·eta (subnormal stop)
  var sigma = sigma0
  var t = offset
  var tPrev = offset
  var tauLast = T(0)
  while true:
    tauLast = extractVector(sigma, p)
    tPrev = t
    t = t + tauLast
    let thresh = pow2i[T](2 * m) * eps * sigma         # 2^(2M)·eps·sigma
    if abs(t) >= thresh or sigma <= sigmaFloor:
      let tau2 = tauLast - (t - tPrev) # FastTwoSum(tPrev, tauLast): exact at stop
      return (t, tau2)
    sigma = sigma * pow2i[T](m) * eps # sigma_{m} = 2^M·eps·sigma_{m-1}

func rumpTransformK[T: SomeFloat](p: var seq[T], offset: T): T =
  ## Rump `TransformK` (Part II, Alg 6.2): a faithful rounding of
  ## `sum(p) + offset` — `fl(tau1 + (tau2 + sum(p')))` after `rumpTransform`. By
  ## Lemma 6.3 the result is `0` iff the exact `sum(p) + offset` is zero, the
  ## property `nearSum` uses for exact-tie detection at the rounding midpoint.
  ## `p` is overwritten with the residual. Returns `Inf` on a `sigma0` overflow.
  let (tau1, tau2) = rumpTransform(p, offset)
  if not isFin(tau1):
    return T(Inf) # sentinel for the sigma0-overflow path
  result = tau1 + (tau2 + naiveSum(p))

func rumpExtract[T: SomeFloat](x: openArray[T], p: var seq[T]): tuple[
    done: bool, val: T, tau1: T, tau2: T] =
  ## Shared front of `accSum` and `nearSum`: the empty, single-element,
  ## non-finite, and `sigma0`-overflow cases all return `done = true` with the
  ## final `val` (0, `x[0]`, or the `superSum` fallback). Otherwise `x` is copied
  ## into `p`, `rumpTransform(p, 0)` runs, and the residual `p` (in place) with
  ## the `(tau1, tau2)` split are returned for the caller's result-specific tail.
  ## `p` is meaningful only when `done = false`.
  if x.len == 0:
    return (true, T(0), T(0), T(0))
  if x.len == 1:
    return (true, x[0], T(0), T(0))
  if not allFin(x):
    return (true, superSum(x), T(0), T(0))
  p = newSeq[T](x.len)
  for i in 0 ..< x.len:
    p[i] = x[i]
  let (tau1, tau2) = rumpTransform(p, T(0))
  if not isFin(tau1): # sigma0-overflow or mid-accumulation overflow
    return (true, superSum(x), T(0), T(0))
  return (false, T(0), tau1, tau2)

func accSum*[T: SomeFloat](x: openArray[T]): T {.contractual.} =
  ## Rump `AccSum` (Part I, Alg 4.5): a *faithful* rounding of `sum(x)` — no
  ## float lies between the result and the exact sum, so `|result - sum(x)| <
  ## 2·eps·ufp(result)` (the 1-ulp bound). Built on the `Transform` repeat-until
  ## over `ExtractVector`; runtime grows with `log2(cond(sum(x)))`.
  ##
  ## Non-finite input or a `sigma0` that overflows the exponent range (input
  ## near the float max, where the sum itself overflows) falls back to
  ## `superSum`, which is correctly-rounded and so still faithful — the
  ## finite-input-never-NaN contract holds either way.
  ensure:
    x.len != 0 or result == T(0)
    not allFin(x) or classify(result) != fcNan # finite input ⇒ no NaN
  body:
    var p: seq[T]
    let ex = rumpExtract(x, p)
    if ex.done:
      return ex.val
    result = ex.tau1 + (ex.tau2 + naiveSum(p)) # p is the residual p'

func nearSum*[T: SomeFloat](x: openArray[T]): T {.contractual.} =
  ## Rump `NearSum` (Part II, Alg 7.4): the *correctly rounded* sum — the
  ## IEEE-754 round-to-nearest of the exact `sum(x)`, bit-for-bit. Runs
  ## `Transform` once for a faithful result and the residual `p'`, then resolves
  ## the rounding direction with a second `TransformK(p', R - delta')` offset to
  ## the midpoint between the two candidate floats (Lemma 7.3): its sign picks
  ## `pred`/`res`/`succ`, and the zero case is the exact tie, rounded by
  ## `fl(res + delta')` with ties-to-even.
  ##
  ## The fallback and contract match `accSum`: non-finite input or a
  ## `sigma0` overflow delegates to `superSum` (also correctly rounded). The
  ## correctly-rounded guarantee holds under `2^(2M)·eps <= 1` (`M =
  ## ceil(log2(n+2))`): `n <= 6.7e7` (float64), `n <= 4094` (float32).
  ensure:
    x.len != 0 or result == T(0)
    not allFin(x) or classify(result) != fcNan # finite input ⇒ no NaN
  body:
    var p: seq[T]
    let ex = rumpExtract(x, p)
    if ex.done:
      return ex.val
    let tau1 = ex.tau1
    let tau2 = ex.tau2
    let tau20 = tau2 + naiveSum(p) # fl(tau2 + sum(p'))
    let res = tau1 + tau20 # FastTwoSum(tau1, tau20): |tau1| >= |tau20|
    let delta = tau20 - (res - tau1) # res + delta = tau1 + tau20 exactly
    if delta == T(0):
      return res # the exact sum is a float
    let r = tau2 - (res - tau1) # R = tau2 - Delta; sum(x) - res = R + sum(p') exactly
    when T is float64:
      const eta = cast[float64](1'u64) # 2^-1074
    else:
      const eta = cast[float32](1'u32) # 2^-149
    if delta < T(0): # fl(S) ∈ {pred(res), res}
      let gamma = prevFloat(res) - res # < 0
      if gamma == -eta:
        return res # no predecessor in F (res at the exponent floor)
      let dp = gamma / T(2) # midpoint toward pred(res)
      let d2 = rumpTransformK(p, r - dp) # faithful(S - midpoint); 0 iff exact tie
      if not isFin(d2):
        return superSum(x)
      if d2 > T(0):
        return res
      elif d2 < T(0):
        return prevFloat(res)
      else:
        return res + dp # exact tie: fl(midpoint) rounds to-even
    else: # delta > 0: fl(S) ∈ {res, succ(res)}
      let gamma = nextFloat(res) - res # > 0
      if gamma == eta:
        return res
      let dp = gamma / T(2)
      let d2 = rumpTransformK(p, r - dp)
      if not isFin(d2):
        return superSum(x)
      if d2 > T(0):
        return nextFloat(res)
      elif d2 < T(0):
        return res
      else:
        return res + dp

func conditionNumber*[T: SomeFloat](x: openArray[T]): T {.contractual.} =
  ## Condition number of the sum, `cond(sum) = sum|x_i| / |sum(x_i)|` (Higham
  ## 2002, §4.1; the ORO/Rump bounds are in this quantity). `1` for a
  ## no-cancellation sum, large under catastrophic cancellation, `Inf` when the
  ## sum is exactly `0` (the sum is then ill-conditioned), `0` for empty input.
  ##
  ## The numerator is a plain sum of absolutes (no cancellation, so accurate to
  ## `gamma_{n-1}` without compensation); the denominator uses `neumaierSum`
  ## (magnitude-robust) so a near-zero sum is detected reliably. A numerator
  ## that overflows the float range (`Σ|x_i|` past the max) signals an
  ## ill-conditioned sum and returns `Inf` rather than `Inf/Inf = NaN`.
  ensure:
    not allFin(x) or classify(result) != fcNan # finite input ⇒ no NaN
  body:
    if x.len == 0:
      return T(0)
    var s = T(0)
    for v in x:
      s += abs(v)
    if classify(s) == fcInf:
      return T(Inf) # Σ|x_i| overflowed: the exact ratio is not representable
    let denom = abs(neumaierSum(x))
    if denom == T(0):
      return T(Inf)
    result = s / denom
