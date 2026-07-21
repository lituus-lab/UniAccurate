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
import std/sequtils
import contracts
import ../twosum
import compensatedsum
import naivesum
import exactsum

func transform[T: SomeFloat](x: openArray[T]): tuple[res: T, err: seq[T]] =
  ## ORO Vecsum (Alg 4.6): error-free transform of a sum. Chains `twoSum`
  ## across `x` so that `res + sum(err) == sum(x)` exactly in real arithmetic,
  ## with `res = fl(Σ xᵢ)` the naive sum and `err` the `n-1` rounding errors.
  ## A non-finite operand degrades that step to naive IEEE propagation (the
  ## EFT is lost, the error slot is set to 0); a step whose sum overflows
  ## produces a NaN error, which the cascade's `isFin` guard localizes.
  let n = x.len
  if n == 0:
    return (T(0), newSeq[T](0))
  result.res = x[0]
  result.err = newSeq[T](n - 1)
  for i in 1 ..< n:
    if not isFin(result.res) or not isFin(x[i]):
      result.res = result.res + x[i]
      result.err[i - 1] = T(0)
    else:
      let (s, e) = twoSum(result.res, x[i])
      result.res = s
      result.err[i - 1] = e

func sum2*[T: SomeFloat](x: openArray[T]): T {.contractual.} =
  ## ORO `sum2` (Alg 4.1) — the magnitude-robust compensated sum. Identical in
  ## value to `neumaierSum`: the ORO Vecsum-with-final-correction is Neumaier's
  ## scheme. Exposed under the ORO name for API symmetry with `sumK`.
  ##
  ## Forward bound: `|fl(sum2) - Σxᵢ| <= (2u + O(nu²)) · Σ|xᵢ|`. See `neumaierSum`
  ## for the magnitude-robustness discussion (recovers a small addend dominated
  ## by the running sum where `kahanSum` loses it).
  ensure:
    x.len != 0 or result == T(0)
    not allFin(x) or classify(result) != fcNan # finite input ⇒ no NaN
  body:
    result = neumaierSum(x)

func sumK*[T: SomeFloat](x: openArray[T], K: int): T {.contractual.} =
  ## ORO `SumK` (Alg 4.8): K-fold cascaded compensated summation. Applies the
  ## error-free `transform` K times, summing the partial results: K=1 is the
  ## naive `twoSum` chain (≈ `naiveSum`), K=2 the first-order compensated sum
  ## (≈ `neumaierSum`), K=3 the second-order (≈ `kleinSum`), and each further
  ## transform compensates one order of rounding error.
  ##
  ## Forward bound (ORO Thm 4.9): `|fl(sumK) - Σxᵢ| <= γ_{n-1}^K · Σ|xᵢ| /
  ## (1 - γ_{n-1}^K)`, so the error is `(n·u)^K · Σ|xᵢ|`. `K` is the cascade
  ## depth (K >= 1; K < 1 is treated as 1).
  ensure:
    x.len != 0 or result == T(0)
    not allFin(x) or classify(result) != fcNan # finite input ⇒ no NaN
  body:
    if x.len == 0:
      return T(0)
    if x.len == 1:
      return x[0]
    let k = if K < 1: 1 else: K
    var res: T
    var err: seq[T]
    (res, err) = transform(x)
    result = res
    for _ in 1 ..< k:
      if err.len == 0:
        break
      (res, err) = transform(err)
      # An overflow poisons the error vector (a `twoSum` whose sum overflows
      # yields a NaN error), so a later partial `res` can be NaN/Inf. Adding it
      # to a running ±Inf would evaluate Inf - Inf = NaN; the true finite sum
      # is unrecoverable past the float range anyway, so skip the compensation
      # once the running sum overflowed (±Inf is the correctly-signed round).
      if isFin(result):
        result += res

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
  ## The next representable float strictly greater than `x` (toward +Inf).
  when T is float64:
    let b = cast[uint64](x)
    if b == 0x8000_0000_0000_0000'u64: # -0.0 -> smallest positive subnormal
      return cast[float64](1'u64)
    return cast[float64](b + 1'u64)
  else:
    let b = cast[uint32](x)
    if b == 0x8000_0000'u32:
      return cast[float32](1'u32)
    return cast[float32](b + 1'u32)

func prevFloat[T: SomeFloat](x: T): T =
  ## The previous representable float strictly less than `x` (toward -Inf).
  when T is float64:
    let b = cast[uint64](x)
    # Both zeros round to the smallest negative subnormal (bit decrement would
    # cross the sign boundary into a signalling-NaN bit pattern).
    if b == 0'u64 or b == 0x8000_0000_0000_0000'u64:
      return cast[float64](0x8000_0000_0000_0001'u64)
    return cast[float64](b - 1'u64)
  else:
    let b = cast[uint32](x)
    if b == 0'u32 or b == 0x8000_0000'u32:
      return cast[float32](0x8000_0001'u32)
    return cast[float32](b - 1'u32)

func extractVector[T: SomeFloat](sigma: T, p: openArray[T]): tuple[
    tau: T, pPrime: seq[T]] =
  ## Rump `ExtractVector` (Part I, Alg 3.4): with `sigma` a power of two `>=
  ## 2^M·max|p_i|`, splits each `p_i` into a high part `q_i` (absorbed into the
  ## running `tau`) and a residual `p'_i = p_i - q_i`, so `p_i = q_i + p'_i`
  ## exactly and `sum(p) = tau + sum(p')`. The high parts need not be stored.
  result.tau = T(0)
  result.pPrime = newSeq[T](p.len)
  for i in 0 ..< p.len:
    let qi = (sigma + p[i]) - sigma
    result.pPrime[i] = p[i] - qi
    result.tau = result.tau + qi

func rumpTransform[T: SomeFloat](p: openArray[T]): tuple[tau1: T, tau2: T,
    pPrime: seq[T]] =
  ## Rump `Transform` (Part I, Alg 4.1/4.4): the faithful error-free transform.
  ## Produces `(tau1, tau2)` (a `FastTwoSum` split of the high-order sum) and a
  ## residual vector `pPrime` with `sum(p) = tau1 + tau2 + sum(pPrime)` exactly
  ## in real arithmetic, `max|pPrime| <= eps·sigma`. `M = ceil(log2(n+2))`;
  ## `eps` is the machine epsilon. The repeat-until shrinks `sigma` by
  ## `2^M·eps` until the accumulated high part `t` dominates the residual.
  let n = p.len
  if n == 0:
    return (T(0), T(0), newSeq[T](0))
  var mu = T(0)
  for v in p:
    let a = abs(v)
    if a > mu:
      mu = a
  if mu == T(0):
    return (T(0), T(0), newSeq[T](n))
  when T is float64:
    const eps = cast[float64](0x3CB0_0000_0000_0000'u64) # 2^-52 (machine epsilon)
    const eta = cast[float64](1'u64) # 2^-1074 (smallest subnormal)
  else:
    const eps = cast[float32](0x3400_0000'u32) # 2^-23
    const eta = cast[float32](1'u32) # 2^-149
  let m = ceilLog2(n + 2) # M = ceil(log2(n+2))
  let sigma0 = nextPowerTwo(mu) * pow2i[T](m)             # 2^(M + ceil(log2 mu))
                                                          # sigma0 overflows only when mu is near the top of the range, i.e. the sum
                                                          # itself overflows; the caller falls back to the superaccumulator in that case.
  if not isFin(sigma0):
    return (T(Inf), T(0), newSeq[T](0))
  let sigmaFloor = T(0.5) * (T(1) / eps) * eta # 0.5·eps^-1·eta (subnormal stop)
  var sigma = sigma0
  var t = T(0)
  var tPrev = T(0)
  var tauLast = T(0)
  var pCur = newSeq[T](n)
  for i in 0 ..< n:
    pCur[i] = p[i]
  var pFinal: seq[T]
  while true:
    let (tau, pNext) = extractVector(sigma, pCur)
    tPrev = t
    t = t + tau
    tauLast = tau
    let thresh = pow2i[T](2 * m) * eps * sigma         # 2^(2M)·eps·sigma
    if abs(t) >= thresh or sigma <= sigmaFloor:
      pFinal = pNext
      break
    sigma = sigma * pow2i[T](m) * eps # sigma_m = 2^M·eps·sigma_{m-1}
    pCur = pNext
  # An overflow during accumulation (t -> Inf) breaks the error-free property;
  # signal it so the caller falls back to the superaccumulator.
  if not isFin(t):
    return (T(Inf), T(0), newSeq[T](0))
  # (tau1, tau2) = FastTwoSum(t^{m-1}, tau^{m}); twoSum is value-identical and
  # needs no |a| >= |b| precondition (the operands are finite: bounded by sigma0).
  let (tau1, tau2) = twoSum(tPrev, tauLast)
  result = (tau1, tau2, pFinal)

func faithfulSum[T: SomeFloat](x: openArray[T]): T =
  ## Rump `AccSum` core (Part I, Alg 4.5): `tau1 + (tau2 + sum(pPrime))`, the
  ## faithful rounding of `sum(x)`. Assumes finite `x` with a non-overflowing
  ## `sigma0` — the public `accSum` guards both and falls back otherwise.
  let (tau1, tau2, pPrime) = rumpTransform(x)
  if not isFin(tau1):
    return T(Inf) # sentinel for the sigma0-overflow path
  result = tau1 + (tau2 + naiveSum(pPrime))

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
    if x.len == 0:
      return T(0)
    if not allFin(x):
      return superSum(x)
    let (tau1, tau2, pPrime) = rumpTransform(x)
    if not isFin(tau1): # sigma0-overflow or mid-accumulation overflow
      return superSum(x)
    result = tau1 + (tau2 + naiveSum(pPrime))

func nearSum*[T: SomeFloat](x: openArray[T]): T {.contractual.} =
  ## Rump `NearSum` (Part II, Alg 7.4): the *correctly rounded* sum — the
  ## IEEE-754 round-to-nearest of the exact `sum(x)`, bit-for-bit. Runs
  ## `Transform` for a faithful result and residual, then resolves the rounding
  ## direction with a second faithful sum of the residual offset to the
  ## midpoint between the two candidate floats.
  ##
  ## The fallback and contract match `accSum`: non-finite input or a
  ## `sigma0` overflow delegates to `superSum` (also correctly rounded).
  ensure:
    x.len != 0 or result == T(0)
    not allFin(x) or classify(result) != fcNan # finite input ⇒ no NaN
  body:
    if x.len == 0:
      return T(0)
    if not allFin(x):
      return superSum(x)
    let (tau1, tau2, pPrime) = rumpTransform(x)
    if not isFin(tau1): # sigma0-overflow → exact fallback
      return superSum(x)
    when T is float64:
      const eta = cast[float64](1'u64) # 2^-1074
    else:
      const eta = cast[float32](1'u32) # 2^-149
    let tau2p = tau2 + naiveSum(pPrime)
    let (res, delta) = twoSum(tau1, tau2p) # res + delta = tau1 + tau2p faithfully
    if delta == T(0):
      return res # the exact sum is a float
    let r = tau2 - (res - tau1) # s - res = r + sum(pPrime), exactly
    if delta < T(0): # fl(s) ∈ {pred(res), res}
      let gamma = prevFloat(res) - res # < 0
      if gamma == -eta:
        return res # no predecessor in F (res at the negative exponent floor)
      let dp = gamma / T(2) # midpoint toward pred(res)
      let d2 = faithfulSum(concat(@[r - dp], pPrime)) # sign of s - midpoint
      if d2 > T(0):
        return res
      elif d2 < T(0):
        return prevFloat(res)
      else:
        return res + dp # exactly the midpoint
    else: # delta > 0: fl(s) ∈ {res, succ(res)}
      let gamma = nextFloat(res) - res # > 0
      if gamma == eta:
        return res
      let dp = gamma / T(2)
      let d2 = faithfulSum(concat(@[r - dp], pPrime))
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
  ## (magnitude-robust) so a near-zero sum is detected reliably.
  ensure:
    not allFin(x) or classify(result) != fcNan # finite input ⇒ no NaN
  body:
    if x.len == 0:
      return T(0)
    var s = T(0)
    for v in x:
      s += abs(v)
    let denom = abs(neumaierSum(x))
    if denom == T(0):
      return T(Inf)
    result = s / denom
