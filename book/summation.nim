# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniAccurate

nbInit(theme = useNimibook)
useLituus()
nb.title = "Summation"

nbText: """
## Summation

The EFT layer recovers the per-operation error; summation algorithms trade
cost for how much that error accumulates over `n` additions. Three variants,
ordered by worst-case rounding depth (deepest first):

- `naiveSum` — left-to-right; `n - 1` additions, error `O(n)`.
- `pairwiseSum` — recursive divide-and-conquer with a naive base case of
  `PairwiseThreshold = 128`; worst-case error `O(log n)`.
- `pairwiseSumIterative` — bottom-up pair combine, no recursion; the tightest
  pairwise bound, `⌈log₂ n⌉ · u · Σ|xᵢ|`.

With `u = ε/2` the unit roundoff (IEEE-754 roundTiesToEven), the forward error
bounds are, to first order (ignoring `O(u²)`),

    naive:               (n - 1)                 · u · Σ|xᵢ|
    pairwise (base b):   ((b - 1) + ⌈log₂(n/b)⌉) · u · Σ|xᵢ|
    iterative:           ⌈log₂ n⌉                · u · Σ|xᵢ|

so pairwise replaces the linear `n - 1` rounding depth with a logarithmic
one. The same three functions cross the other surfaces: the C ABI exposes
`ua_sum_naive`, `ua_sum_pairwise`, `ua_sum_pairwise_iterative`, and Python
exposes `naive_sum`, `pairwise_sum`, `pairwise_sum_iterative`.
"""

nbCode:
  let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  echo "naiveSum             = ", naiveSum(xs)
  echo "pairwiseSum          = ", pairwiseSum(xs)
  echo "pairwiseSumIterative = ", pairwiseSumIterative(xs)
  echo "PairwiseThreshold    = ", PairwiseThreshold

nbText: """
All three return `55.0` here — integer-valued data sums exactly at every
precision. On data that rounds, the worst case is far gentler for pairwise and
iterative: the rounding depth is logarithmic rather than linear, so fewer
roundings touch a small addend on its way to the total. A specific input is
not guaranteed to come out closer to the exact sum — the per-instance order
depends on the data — but the worst-case bound is.

**Limitation.** Every variant propagates NaN/Inf and can yield `NaN` from
opposite-sign overflow (`+Inf + -Inf = NaN`, IEEE-754), at a merge node as
well as within a base block. No fallback is applied: an exact
superaccumulator path is deferred to a later algorithm. Finite inputs that do
not overflow stay finite.

### References

- Higham, N.J. (1993). "The accuracy of floating point summation using
  pairwise summation". *SIAM J. Sci. Comput.* 14(4), 783–799.
  doi:10.1137/0914050 — worst-case `O(log n)` bound for pairwise.
- Higham, N.J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd
  ed., §4.2. SIAM. ISBN 978-0-89871-521-7 — the `(n - 1) · u · Σ|xᵢ|` naive
  bound and the block-pairwise analysis.
- Tasche, D., Zeuner, H. (2000). "Worst and average case roundoff error
  analysis in floating point summation". In: *Handbook of
  Analytic-Computational Methods in Applied Mathematics*, ch. 8, CRC Press —
  RMS `O(√log n)` error for pairwise versus `O(√n)` for naive (random-sign
  data).
- NumPy pairwise tier: `numpy/_core/src/umath/loops.c.src`,
  `PW_BLOCKSIZE = 128`, PR #3685 (Taylor 2013).
"""

nbText: """
## Compensated summation

Pairwise shrinks the *worst-case* rounding depth; compensation shrinks the
*per-step* error itself. Each addition's rounding error is recovered by an
error-free transform and fed back into the sum, so the forward error's
*leading-order* term no longer grows with `n` (the residual `O(nε²)` still
accumulates):

- `kahanSum` — Kahan's single-compensation scheme, 4 FLOPs/term.
- `neumaierSum` — Kahan-Babuška-Neumaier, the magnitude-robust variant, 7
  FLOPs/term (built on `twoSum`).
- `kleinSum` — Klein's two-level scheme, ~13 FLOPs/term, ~ε² accuracy.

With `ε` the machine epsilon (`2^-52` for float64), the forward bounds (first
order) are

    kahanSum / neumaierSum:  (2ε + O(nε²)) · Σ|xᵢ|
    kleinSum:                O(ε²) · Σ|xᵢ|

so compensation trades a few extra FLOPs per term for a leading-order error
independent of `n` (the `O(nε²)` tail still grows, but dominates only for huge
`n·ε`). The bound is in `Σ|xᵢ|` (the input magnitude), not `|S|`:
cancellation (`Σ|xᵢ| >> |S|`) still grows the *relative* error — compensation
bounds the absolute error, the best attainable without sorting.
"""

nbCode:
  let ys = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  echo "naiveSum    = ", naiveSum(ys)
  echo "kahanSum    = ", kahanSum(ys)
  echo "neumaierSum = ", neumaierSum(ys)
  echo "kleinSum    = ", kleinSum(ys)
  # Kahan loses a small addend dominated by the running sum; Neumaier/Klein
  # recover it. The exact sum is 2.0.
  let mag = [1.0, 1e100, 1.0, -1e100]
  echo "kahanSum    (magnitude) = ", kahanSum(mag)
  echo "neumaierSum (magnitude) = ", neumaierSum(mag)
  echo "kleinSum    (magnitude) = ", kleinSum(mag)

nbText: """
On `0.1·10` the naive sum drops the last bit (`0.9999999999999999`); all three
compensated sums recover `1.0`. On the magnitude case the exact sum is `2.0`:
`kahanSum` loses the small `1`s (they fall below the ulp of `1e100`), while
`neumaierSum` and `kleinSum` recover `2.0` — the branchless `twoSum` handles
both operand orderings, which is the magnitude-robustness fix.

The same three functions cross the other surfaces: the C ABI exposes
`ua_sum_kahan`, `ua_sum_neumaier`, `ua_sum_klein`, and Python exposes
`kahan_sum`, `neumaier_sum`, `klein_sum`.

**Limitation.** As with the pairwise sums, every variant propagates NaN/Inf and
can yield `NaN` from opposite-sign overflow (`+Inf + -Inf = NaN`). The per-step
`isFin` guard keeps an `Inf − Inf` from ever being evaluated on finite input,
so finite inputs never yield NaN (overflow gives a single-sign ±Inf); an exact
superaccumulator path is deferred to a later algorithm.

### References

- Kahan, W. (1965). "Pracniques: Further Remarks on Reducing Truncation
  Errors". *Comm. ACM* 8(1), 40. doi:10.1145/363707.363723
- Babuška, I. (1969). "Numerical Stability in Mathematical Analysis". *Proc.
  IFIP Congress 1968*, pp. 11–23. North-Holland.
- Neumaier, A. (1974). "Rundungsfehleranalyse einiger Verfahren zur Summation
  endlicher Summen". *ZAMM* 54(1), 39–51. doi:10.1002/zamm.19740540106
- Klein, A. (2006). "A Generalized Kahan-Babuška-Summation-Algorithm".
  *Computing* 76(3-4), 279–293. doi:10.1007/s00607-005-0139-x
- Higham, N.J. (1993). "The Accuracy of Floating Point Summation". *SIAM J.
  Sci. Comput.* 14(4), 783–799. doi:10.1137/0914050
"""

nbText: """
"""

nbSave
