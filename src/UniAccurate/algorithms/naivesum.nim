# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Naive (sequential) summation.
##
## Accumulates `x` left to right in the working precision, with no
## compensation:
##
##     s_0 = 0
##     s_k = fl(s_{k-1} + x_k),  k = 1..n
##     result = s_n
##
## The baseline against which the pairwise and compensated algorithms are
## measured. Cheapest path — one addition per element, no auxiliary storage —
## and the weakest forward error bound. With `u` the unit roundoff (`ε/2`
## under IEEE-754 roundTiesToEven) and `S = Σ x_i` the exact sum,
##
##     |fl(sum) - S| <= (n - 1) * u * Σ|x_i|   (first order; O(u^2) ignored)
##
## so the absolute error grows linearly in `n` and in the input magnitude.
## Higham (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
## §4.2. SIAM. ISBN 978-0-89871-521-7.
##
## Input domain and limitations
## ============================
##
## Finite floats. A single NaN or ±Inf poisons the result (it propagates
## through the running sum unchanged), matching a plain left fold. The
## opposite-sign overflow artifact `+Inf + -Inf = NaN` (IEEE-754) is a known
## limitation shared with every summation in this layer until the
## superaccumulator fallback lands; see the pairwise module.
import contracts

func naiveSum*[T: SomeFloat](x: openArray[T]): T {.contractual.} =
  ## Left-to-right sum of `x` in the working precision. Empty input is `0`;
  ## `n - 1` additions for `n` elements, starting from `x[0]` so a leading
  ## signed zero (e.g. `-0.0`) is preserved rather than folded to `+0.0`.
  ensure:
    x.len != 0 or result == T(0)
  body:
    if x.len == 0:
      result = T(0)
      return
    result = x[0]
    for i in 1 ..< x.len:
      result += x[i]
