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
