# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Correctly-rounded summation via adaptive-precision expansion.
##
## `shewchukSum` (alias `fsum`) returns the sum of an array rounded once to
## the working precision under round-to-nearest-even — the same guarantee as
## Python's `math.fsum`, whose algorithm this implements. It is the strongest
## accuracy a fixed-precision sum can offer: the result is the exact real sum
## `S = Σ xᵢ` after a single rounding.
##
## Algorithm
## =========
##
## Two phases over a non-overlapping expansion of partials whose magnitudes
## strictly increase (Shewchuk's fast-expansion sum):
##
## 1. **Accumulation.** Each addend is merged into the partials with `twoSum`.
##   The error of `twoSum(hi, p[j])` replaces `p[j]` (kept: a non-overlapping
##   low bit), and the rounded high part propagates as the running `hi` to the
##   next partial. The expansion collects, without rounding loss, every bit the
##   real sum carries.
##
## 2. **Collapse.** From the largest partial downward, `twoSum` folds each
##   smaller partial into the running result; the first *inexact* step (nonzero
##   error `lo`) stops — the remaining partials are below the ulp of the result
##   and cannot affect it. A round-half-to-even adjustment resolves the lone
##   tie straddled by `lo` and the next smaller partial.
##
## The non-overlap and magnitude ordering are the invariants that make one
## left-to-right collapse correctly rounded; Shewchuk (1997) proves them, and
## they hold for the `twoSum` merge exactly as for CPython's swapped
## `fastTwoSum` variant.
##
## Correctness assumptions
## ========================
##
## Inherits `twosum`'s IEEE-754 assumptions (roundTiesToEven, `FLT_EVAL_METHOD
## == 0`, no `-ffp-contract`/`-ffast-math`, finite non-overflowing
## intermediates). Under those the result is correctly rounded and
## order-independent.
##
## Robustness
## ==========
##
## Correct rounding holds only while every partial stays finite. A non-finite
## addend (NaN / ±Inf) or an intermediate overflow invalidates the exact
## expansion, so the function falls back to a naive IEEE sum of the input: NaN
## propagates, ±Inf propagates, `+Inf + -Inf` yields NaN, finite overflow
## yields ±Inf. Finite, non-overflowing inputs never raise; the `twoSum`
## finite-operand precondition is upheld by guarding each step.
##
## Contracts
## =========
##
## `{.contractual.}` (NimContracts, debug-only, compiled away under
## `-d:release`/`-d:danger`). Two postconditions: `[]` sums to zero, and finite
## input ⇒ result is never NaN (overflow ⇒ ±Inf). The finite-input safety
## holds because the naive fallback is sign-stable (a partial sum that becomes
## ±Inf keeps its sign — finite addends cannot flip ±Inf, so `+Inf + -Inf` is
## never evaluated), and the expansion path's round-half-to-even guard
## `y == x - hi` rejects any `±Inf + ∓Inf = NaN`. The correctly-rounded
## guarantee itself is real-arithmetic, verified against a `math.fsum` oracle
## in `tests/test_shewchuksum.nim`, not by a float-level contract.
##
## References
## ==========
##
## - Shewchuk, J.R. (1997). "Adaptive Precision Floating-Point Arithmetic and
##   Fast Robust Geometric Predicates". *Discrete & Computational Geometry*
##   18(3), 305–363. doi:10.1007/PL00009321
## - Hettinger, R. (2005). `math.fsum` (ASPN Cookbook recipe 393090), adopted by
##   CPython; the round-half-to-even collapse and magnitude-ascending merge
##   are that recipe's contribution over Shewchuk's expansion arithmetic.
## - Goldberg, D. (1991). "What Every Computer Scientist Should Know About
##   Floating-Point Arithmetic". *ACM Comput. Surv.* 23(1), 5–48.
##   doi:10.1145/103162.103163
## - Dekker, T.J. (1971). "A Floating-Point Technique for Extending the
##   Available Precision". *Numer. Math.* 18(3), 224–242.
##   doi:10.1007/BF01397083
import std/math
import contracts
import ../twosum
import ./naivesum

func shewchukTotal[T: SomeFloat](partials: openArray[T]): T =
  ## Collapse a non-overlapping, magnitude-ascending expansion into one
  ## correctly-rounded float (the CPython `fsum` final step).
  if partials.len == 0:
    return T(0)
  var n = partials.len - 1
  var hi = partials[n]
  var lo = T(0)
  while n > 0:
    let prev = hi
    dec n
    let (s, e) = twoSum(prev, partials[n]) # |prev| > |p[n]|: error e is exact
    hi = s
    lo = e
    if lo != T(0): # first inexact step: the rest is below ulp(hi)
      break
  # Round-half-to-even across the tie straddled by `lo` and the next smaller
  # partial: if they share sign, `lo` is at least half an ulp of `hi`, so
  # doubling it tests whether `hi + 2·lo` is exact — if so, `hi` was a tie and
  # rounds to even.
  if n > 0 and ((lo < T(0) and partials[n - 1] < T(0)) or
                (lo > T(0) and partials[n - 1] > T(0))):
    let y = lo + lo
    let x = hi + y
    if y == x - hi:
      hi = x
  result = hi

func shewchukSum*[T: SomeFloat](x: openArray[T]): T {.contractual.} =
  ## Correctly-rounded sum via Shewchuk's adaptive-precision expansion: `Σ xᵢ`
  ## rounded once to the working precision under round-to-nearest-even.
  ##
  ## NaN / ±Inf in the input, or an intermediate overflow, abandon the exact
  ## expansion and propagate as a naive IEEE sum (NaN / ±Inf / NaN for
  ## `+Inf + -Inf`). Finite inputs never raise.
  ##
  ## Example:
  ##
  ## .. code-block:: nim
  ##   let s = shewchukSum([1.0, 1e100, 1.0, -1e100])
  ##   doAssert s == 2.0
  ensure:
    x.len != 0 or result == T(0)
    not allFin(x) or classify(result) != fcNan # finite input ⇒ no NaN (overflow ⇒ ±Inf)
  body:
    if x.len == 0:
      return T(0)
    var partials: seq[T] = @[]
    for xi in x:
      if not isFin(xi):
        return naiveSum(x) # non-finite addend: IEEE propagation
      var hi = xi
      var i = 0
      for j in 0 ..< partials.len:
        let (s, e) = twoSum(hi, partials[j])
        if e != T(0):
          partials[i] = e
          inc i
        hi = s
        if not isFin(hi):
          return naiveSum(x) # intermediate overflow: IEEE propagation
      partials.setLen(i)
      if hi != T(0):
        partials.add(hi)
    result = shewchukTotal(partials)

func fsum*[T: SomeFloat](x: openArray[T]): T {.contractual, inline.} =
  ## Alias of `shewchukSum`, mirroring Python's `math.fsum`. Inherits the
  ## finite-input safety postcondition.
  ensure:
    x.len != 0 or result == T(0)
    not allFin(x) or classify(result) != fcNan # finite input ⇒ no NaN (overflow ⇒ ±Inf)
  body:
    shewchukSum(x)
