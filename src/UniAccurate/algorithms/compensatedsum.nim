# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Compensated summation.
##
## Compensated (error-compensating) summation recovers the low-order bits lost
## to rounding at each `+` by tracking the per-step rounding error with an
## error-free transformation and feeding it back into the sum. Where `naiveSum`
## lets the forward error grow as `(n - 1)·u·Σ|xᵢ|` (linear in `n`),
## compensation holds it to a constant times `Σ|xᵢ|` — the error stops growing
## with the number of terms.
##
##   * `kahanSum` — Kahan's single-compensation scheme (4 FLOPs/term, O(1)
##     space). Faithful to Kahan's original form: the compensation `c` is folded
##     into the next addend at each step (`y = v - c`), with no final `+ c`.
##
## Error bound
## ===========
##
## Let `S = Σ xᵢ` be the exact sum, `Σ|xᵢ|` the L1 magnitude, and `ε` the machine
## epsilon (`2^-52` for float64, `2^-23` for float32). Under the IEEE-754
## assumptions of `twosum` (roundTiesToEven, `FLT_EVAL_METHOD == 0`, no
## `-ffast-math`, finite non-overflowing intermediates):
##
##     |fl(kahanSum) - S| <= (2ε + O(nε²)) · Σ|xᵢ|
##
## The bound is in `Σ|xᵢ|`, not `|S|`: for catastrophic cancellation
## (`Σ|xᵢ| >> |S|`) the *absolute* error stays small but the *relative* error
## grows — compensated summation bounds the absolute error against the input
## magnitude, the best attainable without sorting. For inputs where a later
## addend dominates the running sum (e.g. `1, 1e100, 1, -1e100`), Kahan's form
## loses the small addend; a magnitude-robust variant (a later module) handles
## that. Higham (1993), "The Accuracy of Floating Point Summation".
##
## Correctness assumptions
## ========================
##
## As for `twosum`: `roundTiesToEven`, `FLT_EVAL_METHOD == 0`, no `-ffast-math`.
## These functions accept arbitrary arrays (including NaN/±Inf): a non-finite
## element or a partial sum that overflows degrades that step to a naive IEEE
## `+`, so NaN/Inf propagate as a plain sum would. Finite inputs never raise; the
## result may be ±Inf only on genuine overflow, never NaN (the per-step `isFin`
## guard keeps an `Inf − Inf = NaN` from ever being evaluated). The `twoSum`
## finite-operand precondition is upheld by guarding each step.
##
## Contracts
## =========
##
## `{.contractual.}` (NimContracts, debug-only, compiled away under
## `-d:release`/`-d:danger`). Two postconditions: the trivial empty-input case
## (`[]` sums to zero), and the safety property — under all-finite input the
## result is never NaN. The accuracy bound above is real-arithmetic and is
## verified against an exact oracle in the test suite, not by a float-level
## contract.
##
## References
## ==========
##
## - Kahan, W. (1965). "Pracniques: Further Remarks on Reducing Truncation
##   Errors". *Comm. ACM* 8(1), 40. doi:10.1145/363707.363723
## - Higham, N.J. (1993). "The Accuracy of Floating Point Summation".
##   *SIAM J. Sci. Comput.* 14(4), 781–799. doi:10.1137/0914050
## - Higham, N.J. (2002). *Accuracy and Stability of Numerical Algorithms*,
##   2nd ed., §4.3. SIAM. ISBN 978-0-89871-521-7
import std/math
import contracts
import ../twosum

func kahanSum*[T: SomeFloat](x: openArray[T]): T {.contractual.} =
  ## Kahan (1965) compensated summation: tracks the rounding error of each `+`
  ## in a running compensation `c` and subtracts it from the next addend,
  ## recovering the lost low-order bits. 4 FLOPs/term, O(1) space.
  ##
  ## Faithful to Kahan's original form (the compensation is folded in at each
  ## step via `y = v - c`; no final `+ c`). For inputs where a later addend
  ## dominates the running sum, a magnitude-robust variant does better.
  ##
  ## Example:
  ##
  ## .. code-block:: nim
  ##   let s = kahanSum([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
  ##   doAssert s == 1.0
  ensure:
    x.len != 0 or result == T(0)
    not allFin(x) or classify(result) != fcNan # finite input ⇒ no NaN
  body:
    if x.len == 0:
      return T(0)
    result = x[0]
    var c = T(0)
    for i in 1 ..< x.len:
      let v = x[i]
      if not isFin(result) or not isFin(v):
        result = result + v # EFT lost; naive IEEE propagation
        continue
      let y = v - c # addend corrected by the prior step's error
      let t = result + y
      c = (t - result) - y # algebraically 0; captures this step's error
      result = t
