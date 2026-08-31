# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniAccurate

nbInit(theme = useNimibook)
useLituus()
nb.title = "Statistical reductions"

nbText: """
## Statistical reductions

`scaledEuclideanNorm` applies the BLAS LASSQ scaling recurrence: intermediate
squares cannot overflow or underflow when the final norm is representable.
`scaledMean` uses the exact rounded total when representable and exactly
accumulates represented `x[i] / n` quotients only as an overflow fallback. It
does not claim correctly rounded rational division of the exact sum.
`centeredCosineSimilarity` normalizes deviations
before its exact dot product, keeping correlation-like results representable.
`centeredProjectionCoefficient` additionally combines binary scale exponents,
so a least-squares slope remains representable when both raw moments overflow.
`centeredSumSquares` and `centeredCrossProduct` subtract caller-supplied centers,
then accumulate the represented products with the exact superaccumulator.
Estimator conventions remain the responsibility of the statistical layer.
"""

nbCode:
  let observations = [1e10, 1e10 + 1.0, 1e10 + 2.0]
  echo "scaled mean max = ", scaledMean([1e308, 1e308])
  echo "scaled norm 3-4 = ", scaledEuclideanNorm([3.0, 4.0])
  echo "centered squares = ", centeredSumSquares(observations, 1e10 + 1.0)
  echo "centered product = ", centeredCrossProduct(observations,
    [2.0, 4.0, 6.0], 1e10 + 1.0, 4.0)
  echo "centered cosine = ", centeredCosineSimilarity(
    [1e308, -1e308], [1e308, -1e308], 0.0, 0.0)
  echo "projection = ", centeredProjectionCoefficient(
    [1e308, 0.0, -1e308], [1e308, 0.0, -1e308], 0.0, 0.0)

nbText: """
The C ABI exposes `ua_sum_oro` / `ua_sum_acc` / `ua_sum_near` / `ua_sum_k` /
`ua_condition_number` and Python exposes `oro_sum` / `acc_sum` / `near_sum` /
`sum_k` / `condition_number`.

### References

- Ogita, T., Rump, S.M., Oishi, S. (2005). "Accurate Sum and Dot Product".
  *SIAM J. Sci. Comput.* 26(6), 1950–1988. doi:10.1137/S0036142903448029 —
  the `transform` (Vecsum, Alg 4.6), `sum2` (Alg 4.1), `sumK` (Alg 4.8).
- Rump, S.M. (2008). "Accurate summation". *Numer. Math.* 110, 385–404.
  doi:10.1007/s00211-008-0144-z — `accSum` (Alg 4.5), `nearSum` (Alg 7.4),
  `ExtractVector` / `Transform` / `TransformK`, Lemmas 6.3 and 7.3.
- Higham, N.J. (2002). *Accuracy and Stability of Numerical Algorithms*,
  2nd ed., §4.1–4.3. SIAM. ISBN 978-0-89871-521-7 — `cond(sum)`, the `γ_k`
  forward bounds.
"""

nbText: """
"""

nbSave
