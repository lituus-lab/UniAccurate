# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Pairwise (recursive) summation.
##
## Splits `x` into two halves, sums each recursively, and adds the partial
## sums. The recursion bottoms out in a naive block sum of at most
## `PairwiseThreshold` elements (reusing `naiveSum` via a zero-copy slice). The
## merge tree has depth `ceil(log2(n/b))`, so the worst-case rounding depth is
## `(b - 1) + ceil(log2(n/b))` additions versus naive's `n - 1`. With `u` the
## unit roundoff and `S = Σ x_i` the exact sum,
##
##     |fl(sum) - S| <= ((b - 1) + ceil(log2(n/b))) * u * Σ|x_i|
##                                                    (first order; O(u^2))
##
## i.e. the worst-case error grows as `O(log n)` instead of naive's `O(n)`.
## Higham (1993), "The accuracy of floating point summation using pairwise
## summation", SIAM J. Sci. Comput. 14(4), 783–799. doi:10.1137/0914050.
## For random-sign inputs the RMS error grows as `O(√log n)` rather than
## naive's `O(√n)`; Tasche & Zeuner (2000), in: *Handbook of
## Analytic-Computational Methods in Applied Mathematics*, ch. 8, CRC Press.
##
## The base-case cutoff mirrors NumPy's recursive tier: NumPy sums naively
## for `n < 8`, with an unrolled 8–128 block, and recursively above 128
## (`PW_BLOCKSIZE = 128`); `numpy/_core/src/umath/loops.c.src`, PR #3685
## (Taylor 2013). The unrolled middle tier is a SIMD concern and lives in the
## simd layer; here the base case is the scalar `naiveSum`.
##
## Input domain and limitations
## ============================
##
## Finite floats. NaN/Inf propagate (a single non-finite value poisons the
## result). The opposite-sign overflow artifact `+Inf + -Inf = NaN` (IEEE-754)
## can arise at a merge node (`a + b` with `a = +Inf`, `b = -Inf`) as well as
## within a base block; no fallback is applied here — the superaccumulator
## exact path is deferred. Same limitation as the naive module.
import contracts
import naivesum

const PairwiseThreshold* = 128
  ## Base-case size: sub-arrays of at most this many elements are summed with
  ## `naiveSum`. Matches NumPy's `PW_BLOCKSIZE`. Larger blocks cut recursion
  ## overhead at the cost of a deeper worst-case rounding path.

func pairwise[T: SomeFloat](x: openArray[T], lo, hi: int): T =
  ## Recursive pairwise sum over the inclusive slice `[lo, hi]` of `x`.
  let n = hi - lo + 1
  if n <= PairwiseThreshold:
    return naiveSum(toOpenArray(x, lo, hi))
  let m = lo + n div 2
  result = pairwise(x, lo, m - 1) + pairwise(x, m, hi)

func pairwiseSum*[T: SomeFloat](x: openArray[T]): T {.contractual.} =
  ## Recursive pairwise sum of `x`. Empty input is `0`; the merge tree has
  ## depth `ceil(log2(n/b))` with a `b = PairwiseThreshold` naive base case.
  ensure:
    x.len != 0 or result == T(0)
  body:
    if x.len == 0:
      return T(0)
    result = pairwise(x, 0, x.high)
