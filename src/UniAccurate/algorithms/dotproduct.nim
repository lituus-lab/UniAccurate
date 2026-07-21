# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Dot product: `naiveDot`, `dot2`, `dotK`.
##
## The dot product `xᵀy = Σ xᵢ·yᵢ` is the natural analogue of summation: each
## product `xᵢ·yᵢ` is computed with an error-free transform (`twoProductFMA` →
## `(h, r₁)` with `h + r₁ = xᵢ·yᵢ` exactly), and the per-term rounding errors
## (the product error `r₁` and the sum error `r₂` from `twoSum`) are themselves
## accumulated with a compensated sum. The forward error then shrinks to
## `O(u^K·cond(xᵀy)·|xᵀy|)` — the result is as accurate as if the dot product
## were computed in *K-fold* working precision (twice for `K = 2`), provided
## `(n−1)u < 1`.
##
## Algorithms
## ==========
##
## `dot2` is ORO Algorithm 5.3 with a *naive* error accumulator (the cascaded
## `R = NaiveSum` case) — O(n) time, O(1) memory, online:
##
##   `(h, r₁) = twoProduct(x₀, y₀); e = r₁; s = h`
##   for i ≥ 1: `(h, r₁ᵢ) = twoProduct(xᵢ, yᵢ); (s, r₂) = twoSum(s, h); e = (e + r₁ᵢ) + r₂`
##   `result = s + e`
##
## `dotK` generalizes this to K-fold precision: the same `twoProductFMA` +
## `twoSum` loop, but the per-term product error `r₁` and sum error `r₂` are
## accumulated in an online `(K−1)`-fold cascade — the O(1)-space, single-pass
## reformulation of the `VecSum` distillation (the cascaded `R = Sum_{K−1}`
## case). `K = 1` is the naive dot. `dot2` is kept as the O(1)-memory
## specialization of the doubled-precision path (cf. `sum2` vs `sumK`);
## `dot2(x, y) == dotK(x, y, 2)` bit-for-bit on finite input.
##
## Error bound
## ===========
##
## Let `S = xᵀy`, `E = Σ|xᵢ·yᵢ|`, `u` the unit roundoff, and
## `γ_m(v) = m·v / (1 − m·v)`. Under the IEEE-754 assumptions of `twosum`
## (roundTiesToEven, `FLT_EVAL_METHOD == 0`, no `-ffast-math`, finite
## non-overflowing intermediates) and provided `(n−1)u < 1`:
##
##     |result − S| <= 2u·|S| + 2·(γ_{n+1}(2u))^K · E
##
## The `2u·|S|` term is the unavoidable final rounding to working precision; the
## cascade term is `(n·u)^K · E`. For `K = 2` this is the Graillat Dot2 bound
## (twice working precision); larger `K` gives K-fold precision. The bound is in
## `E`, not `|S|`: cancellation grows the relative error, not the absolute.
##
## Robustness
## ==========
##
## `twoProductFMA` / `twoSum` require finite operands, so on non-finite input
## or on product / running-sum overflow the error-free transform is invalid and
## `dot2` / `dotK` fall back to `naiveDot` (IEEE propagation of NaN/±Inf and
## overflow) — matching the guard convention of the compensated sums. Finite
## inputs never raise. A length mismatch is a contract precondition violation,
## not a raised exception.
##
## One subtlety: a product of two *finite* values can overflow to ±Inf (unlike a
## sum, whose addends are the finite inputs themselves), so opposite-sign ±Inf
## products can combine to `+Inf + −Inf = NaN` — a summation-order artifact,
## since the exact dot of finite values is finite or a correctly-signed ±Inf,
## never NaN. When `naiveDot`'s result is NaN but every input is finite, it falls
## back to `superDot` (the exact superaccumulator dot), which recovers the true
## value (finite, or correctly-signed ±Inf on genuine overflow). The fallback
## runs only on that rare overflow — the NaN test is free on normal data, and the
## all-finite scan happens only then — and `dot2` / `dotK` inherit it via their
## `naiveDot` overflow paths, so the finite-input safety postcondition holds for
## the whole family.
##
## Contracts
## =========
##
## `{.contractual.}` (NimContracts, debug-only, compiled away under
## `-d:release`): the length-match precondition, the empty-input case, and the
## safety property — under all-finite input the result is never NaN. The accuracy
## bound above is real-arithmetic and verified against an exact oracle in the
## test suite, not by a float-level contract.
##
## `assumeFinite: static bool = false` (opt-in on `dot2` / `dotK`) drops the
## per-element finiteness / overflow guards (the `naiveDot` fallback): the caller
## contracts finite input and no product / running-sum overflow, so the EFT
## recurrence runs bare — matching a plain-C compensated dot loop and removing
## the per-element guard tax. NaN / ±Inf inputs or an intermediate overflow then
## produce undefined garbage instead of the fallback; the default (`false`) keeps
## the guard and is safe. Bit-identical to the guarded path on finite
## non-overflowing input.
##
## References
## ==========
##
## - Ogita, T., Rump, S.M., Oishi, S. (2005). "Accurate Sum and Dot Product".
##   *SIAM J. Sci. Comput.* 26(6), 1950–1988. doi:10.1137/S0036142903448029 —
##   Algorithm 5.3 (DotK), Theorem 5.4 (K-fold bound); the
##   `twoProduct`/VecSum substrate.
## - Graillat, D., Langlois, P., Louvet, N. (2006). "Choosing a Twice More
##   Accurate Dot Product Implementation." ICNAAM 2006. HAL hal-01351480 —
##   Dot2FMA, FMA error extraction, the doubled-precision comparison.
## - Graillat, D., Jézéquel, M. (2020). "Tight Interval Inclusions with
##   Compensated Algorithms." *IEEE Trans. Comput.* 69(12), 1774–1783.
##   doi:10.1109/TC.2019.2924005 — CompDot/Dot2 bound
##   `|res − xᵀy| <= 2u|xᵀy| + 2γ²_{n+1}(2u)|x|ᵀ|y|`.
import std/math
import contracts
import ../twosum
import exactsum # superDot: exact fallback on opposite-sign product overflow

func naiveDot*[T: SomeFloat](x, y: openArray[T]): T {.contractual.} =
  ## Naive dot product `Σ xᵢ·yᵢ` (left-to-right accumulation). NaN/±Inf and
  ## overflow propagate per IEEE; finite inputs never raise. Length mismatch
  ## is a contract precondition violation. A product of two finite values can
  ## overflow to ±Inf, so opposite-sign ±Inf products can combine to
  ## `+Inf + −Inf = NaN` (an order artifact — the exact dot of finite values is
  ## finite or correctly-signed ±Inf, never NaN); on that rare overflow the
  ## result is recovered exactly via `superDot`, gated by all-finite input so
  ## IEEE propagation of real NaN/±Inf inputs is preserved.
  require:
    x.len == y.len
  ensure:
    x.len != 0 or result == T(0)
    not (allFin(x) and allFin(y)) or classify(result) != fcNan # finite input ⇒ no NaN
  body:
    if x.len == 0:
      return T(0)
    result = T(0)
    for i in 0 ..< x.len:
      result += x[i] * y[i]
    if classify(result) == fcNan and allFin(x) and allFin(y):
      # Opposite-sign ±Inf products combined to NaN — a summation-order
      # artifact; the exact dot of finite values is finite or correctly-signed
      # ±Inf. Recover it exactly via the superaccumulator. Rare (overflow only):
      # the NaN test is free on normal data, and the all-finite scan runs only
      # then.
      result = superDot(x, y)

func dot2*[T: SomeFloat](x, y: openArray[T],
    assumeFinite: static bool = false): T {.contractual.} =
  ## Compensated dot product at twice working precision (ORO Algorithm 5.3,
  ## `K = 2`; Graillat Dot2FMA). O(n) time, O(1) memory, online. Error
  ## `<= 2u·|xᵀy| + 2·γ²_{n+1}(2u)·Σ|xᵢ·yᵢ|` — twice precision — when
  ## `(n−1)u < 1`. Falls back to `naiveDot` on non-finite input or overflow;
  ## finite inputs never raise. `dot2(x, y) == dotK(x, y, 2)` bit-for-bit.
  ##
  ## `assumeFinite = true` (opt-in) drops the per-element finiteness / overflow
  ## guards (the `naiveDot` fallback) — the caller contracts (via `require:`,
  ## checked under `-d:contracts`) that every element is finite **and** no
  ## product or running sum overflows, so the EFT recurrence runs bare. NaN /
  ## ±Inf inputs or an intermediate overflow then produce undefined garbage
  ## instead of the `naiveDot` fallback; the default (`false`) keeps the guard
  ## and is safe.
  require:
    x.len == y.len
    not assumeFinite or (allFin(x) and allFin(y)) # opt-in ⇒ finite input
  ensure:
    x.len != 0 or result == T(0)
    assumeFinite or (not (allFin(x) and allFin(y)) or classify(result) !=
        fcNan) # finite ⇒ no NaN (guard path only)
  body:
    if x.len == 0:
      return T(0)
    when not assumeFinite:
      if not isFin(x[0]) or not isFin(y[0]):
        return naiveDot(x, y)
    var (s, r1) = twoProductFMA(x[0], y[0])
    when not assumeFinite:
      if not isFin(s):
        return naiveDot(x, y)
    var e = r1
    for i in 1 ..< x.len:
      when not assumeFinite:
        if not isFin(x[i]) or not isFin(y[i]):
          return naiveDot(x, y)
      let (h, r1i) = twoProductFMA(x[i], y[i])
      when not assumeFinite:
        if not isFin(h) or not isFin(r1i): # product overflow → EFT invalid
          return naiveDot(x, y)
      let (s2, r2) = twoSum(s, h)
      when not assumeFinite:
        if not isFin(s2): # running-sum overflow
          return naiveDot(x, y)
      s = s2
      e = (e + r1i) + r2 # left-assoc, matches the ORO recurrence
    result = s + e

# Forward declarations: `dotK` (a `{.contractual.}` body) dispatches to these
# monomorphized helpers, which must be visible before that body is expanded.
func dotKCascade[T: SomeFloat; K: static int](x, y: openArray[T],
    assumeFinite: static bool): T
func dotKCascadeRT[T: SomeFloat](x, y: openArray[T], K: int,
    assumeFinite: static bool): T

func dotK*[T: SomeFloat](x, y: openArray[T], K: int = 2,
    assumeFinite: static bool = false): T {.contractual.} =
  ## K-fold compensated dot product (ORO Algorithm 5.3). `K = 1` is `naiveDot`;
  ## `K >= 2` keeps the running `twoSum` sum `p` of the products `xᵢ·yᵢ` and
  ## accumulates the per-term product error `r₁` and sum error `r₂` in an online
  ## `(K−1)`-fold cascade — the O(1)-space, single-pass reformulation of the
  ## `VecSum` distillation, giving K-fold overall. Error
  ## `<= 2u·|xᵀy| + 2·(γ_{n+1}(2u))^K·Σ|xᵢ·yᵢ|` when `(n−1)u < 1`. `K·n` FLOPs,
  ## O(K) working memory. Same Inf/NaN/overflow fallback and contract as
  ## `dot2`; `dot2(x, y) == dotK(x, y, 2)` bit-for-bit.
  ##
  ## `K = 2` and `K = 3` dispatch to a compile-time-monomorphized cascade (a
  ## `static int` K unrolls the per-element `twoProductFMA`/`twoSum` loop and
  ## stack-allocates the `array[K-1, T]` error cascade); `K > 3` runs the
  ## runtime-K cascade. See `dot2` for the `assumeFinite` opt-in.
  require:
    x.len == y.len
    not assumeFinite or (allFin(x) and allFin(y)) # opt-in ⇒ finite input
  ensure:
    x.len != 0 or result == T(0)
    assumeFinite or (not (allFin(x) and allFin(y)) or classify(result) !=
        fcNan) # finite ⇒ no NaN (guard path only)
  body:
    if x.len == 0:
      return T(0)
    if K < 2:
      return naiveDot(x, y)
    case K
    of 2:
      return dotKCascade[T, 2](x, y, assumeFinite)
    of 3:
      return dotKCascade[T, 3](x, y, assumeFinite)
    else:
      return dotKCascadeRT(x, y, K, assumeFinite)

func dotKCascade[T: SomeFloat; K: static int](x, y: openArray[T],
    assumeFinite: static bool): T =
  ## Online cascaded dotK over a compile-time-known `K` — the monomorphized
  ## fast path of `dotK`: `array[K-1, T]` stack error cascade (no heap alloc)
  ## and an unrolled per-element `twoProductFMA`/`twoSum` recurrence. `K >= 2`.
  ## Same guards, bound and compensated `(r + p).sum()` reduction as `dotK`.
  template foldErr(eval: untyped) =
    var e = eval
    for j in 0 ..< K - 2: # compensated levels of the (K-1)-cascade
      when not assumeFinite:
        if not isFin(rlvl[j]) or not isFin(e):
          return naiveDot(x, y)
      let (s, err) = twoSum(rlvl[j], e)
      when not assumeFinite:
        if not isFin(s):
          return naiveDot(x, y)
      rlvl[j] = s
      e = err
    rlvl[K - 2] += e # naive tail of the (K-1)-cascade
  var rlvl: array[K - 1, T]
  when not assumeFinite:
    if not isFin(x[0]) or not isFin(y[0]):
      return naiveDot(x, y)
  var (h, r1) = twoProductFMA(x[0], y[0])
  when not assumeFinite:
    if not isFin(h) or not isFin(r1):
      return naiveDot(x, y)
  var p = h
  foldErr(r1)
  for i in 1 ..< x.len:
    when not assumeFinite:
      if not isFin(x[i]) or not isFin(y[i]):
        return naiveDot(x, y)
    let (hi, r1i) = twoProductFMA(x[i], y[i])
    when not assumeFinite:
      if not isFin(hi) or not isFin(r1i):
        return naiveDot(x, y)
    let (p2, r2) = twoSum(p, hi)
    when not assumeFinite:
      if not isFin(p2): # running-sum overflow
        return naiveDot(x, y)
    p = p2
    foldErr(r1i)
    foldErr(r2)
  # Compensated `(r + p).sum()` reduction: fold the running product sum `p`
  # into the (K-1)-level error cascade `rlvl` as a final compensated addend,
  # then sum the cascade recursively (symmetric to `sumK`'s `(c + s).sum()`).
  var e = p
  for j in 0 .. K - 2:
    for m in j .. K - 3:
      let (s, err) = twoSum(rlvl[m], e)
      rlvl[m] = s
      e = err
    rlvl[K - 2] += e
    e = rlvl[j]
  result = rlvl[K - 2]

func dotKCascadeRT[T: SomeFloat](x, y: openArray[T], K: int,
    assumeFinite: static bool): T =
  ## Runtime-K fallback of the cascaded dotK (`K > 3` or uncommon K): a heap
  ## `newSeq[T](K-1)` and a looped recurrence. Called by `dotK` for K not in
  ## {1, 2, 3}; semantically identical to `dotKCascade`, only slower.
  template foldErr(eval: untyped) =
    var e = eval
    for j in 0 ..< K - 2:
      when not assumeFinite:
        if not isFin(rlvl[j]) or not isFin(e):
          return naiveDot(x, y)
      let (s, err) = twoSum(rlvl[j], e)
      when not assumeFinite:
        if not isFin(s):
          return naiveDot(x, y)
      rlvl[j] = s
      e = err
    rlvl[K - 2] += e
  var rlvl = newSeq[T](K - 1)
  when not assumeFinite:
    if not isFin(x[0]) or not isFin(y[0]):
      return naiveDot(x, y)
  var (h, r1) = twoProductFMA(x[0], y[0])
  when not assumeFinite:
    if not isFin(h) or not isFin(r1):
      return naiveDot(x, y)
  var p = h
  foldErr(r1)
  for i in 1 ..< x.len:
    when not assumeFinite:
      if not isFin(x[i]) or not isFin(y[i]):
        return naiveDot(x, y)
    let (hi, r1i) = twoProductFMA(x[i], y[i])
    when not assumeFinite:
      if not isFin(hi) or not isFin(r1i):
        return naiveDot(x, y)
    let (p2, r2) = twoSum(p, hi)
    when not assumeFinite:
      if not isFin(p2):
        return naiveDot(x, y)
    p = p2
    foldErr(r1i)
    foldErr(r2)
  var e = p
  for j in 0 .. K - 2:
    for m in j .. K - 3:
      let (s, err) = twoSum(rlvl[m], e)
      rlvl[m] = s
      e = err
    rlvl[K - 2] += e
    e = rlvl[j]
  result = rlvl[K - 2]
