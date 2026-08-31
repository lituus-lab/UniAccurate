# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniAccurate

nbInit(theme = useNimibook)
useLituus()
nb.title = "Correctly rounded, and exact"

nbText: """
## Correctly-rounded summation

Compensation drives the *leading-order* error to zero; `shewchukSum` drives
the *entire* error to a single rounding. It accumulates each addend into a
non-overlapping, magnitude-ascending expansion with `twoSum`, then collapses
that expansion once under round-to-nearest-even — the exact real sum `Σ xᵢ`
rounded to the working precision, for finite non-overflowing input. This is the
strongest accuracy a fixed-precision sum can offer: on such input the result
matches Python's `math.fsum` bit-for-bit on the tested platforms; the
correctness reference is the exact-rational oracle in `py/tests`. `fsum` is an
alias.

The forward bound is the round-to-nearest bound itself, half a unit in the last
place of the rounded result:

    shewchukSum:  ½ ulp(fl(Σ xᵢ))    (finite result; zero is exact, subnormals use the subnormal ulp)

so the error is bounded in `|S|`, not `Σ|xᵢ|`: cancellation does not grow it.
Correct rounding holds only while every partial stays finite; a non-finite
addend or an intermediate overflow abandons the exact expansion and falls back
to a naive IEEE sum — the result is then a correctly-signed ±Inf (or NaN for
`+Inf + −Inf`), not a correctly-rounded finite value. Finite inputs never yield
NaN.
"""

nbCode:
  let zs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  echo "naiveSum     = ", naiveSum(zs)
  echo "shewchukSum = ", shewchukSum(zs)
  echo "fsum         = ", fsum(zs)
  let cancel = [1.0, 1e100, 1.0, -1e100]
  echo "shewchukSum (magnitude) = ", shewchukSum(cancel)

nbText: """
On `0.1·10` the naive sum drops the last bit (`0.9999999999999999`);
`shewchukSum` returns `1.0` — the exact `1.0` rounded once, not a compensated
approximation. On the magnitude case the exact sum is `2.0`, recovered exactly
where `kahanSum` lost it.

The C ABI exposes `ua_sum_shewchuk` and Python exposes `shewchuk_sum` (and
`fsum` is Nim-only). There is no SIMD kernel: the sequential expansion does not
vectorize, so under `-d:simd` the C ABI dispatches this symbol to the scalar
algorithm (documented in ADR-0006).

### References

- Shewchuk, J.R. (1997). "Adaptive Precision Floating-Point Arithmetic and
  Fast Robust Geometric Predicates". *Discrete & Computational Geometry*
  18(3), 305–363. doi:10.1007/PL00009321 — the expansion arithmetic.
- Hettinger, R. (2005). `math.fsum` (ASPN Cookbook recipe 393090), adopted by
  CPython — the round-half-to-even collapse and magnitude-ascending merge over
  Shewchuk's expansion.
- Goldberg, D. (1991). "What Every Computer Scientist Should Know about
  Floating-Point Arithmetic". *ACM Comput. Surv.* 23(1), 5–48.
  doi:10.1145/103162.103163
"""

nbText: """
## Exact summation (the small superaccumulator)

`shewchukSum` is already correctly rounded — `½ ulp(fl(Σ xᵢ))` — holding the
sum as a float-precision expansion whose `twoSum` merge collects every bit
without rounding loss. Its one ceiling is the float range itself: an
intermediate partial that overflows (the running magnitude exceeds the float
range, even when the exact sum is finite) abandons the expansion and falls back
to IEEE propagation, where opposite-sign overflow can yield `+Inf + −Inf = NaN`.
`superSum` drops that ceiling by accumulating in an *exact integer*
superaccumulator (Neal's small superaccumulator): each addend is split into its
exponent chunk and added into a fixed array of `int64` bins covering the whole
float range, carries deferred lazily, and a single final round produces the
float result. The integer sum is exact, commutative, and associative, so the
final rounding is order-invariant and the error is again `½ ulp(fl(Σ xᵢ))` —
but now the accumulator holds the true magnitude across the whole range, and
opposite-sign overflow cancels exactly instead of yielding `+Inf + −Inf = NaN`.

`superDot` reuses the same accumulator with a 64×64→128 product step, so the
dot product `Σ xᵢyᵢ` is held at its true magnitude even when an individual
product overflows the float range — the finite-input NaN fallback the dot
family relies on.

    superSum:  ½ ulp(fl(Σ xᵢ))      order-invariant, exact integer accumulation
    superDot:  ½ ulp(fl(Σ xᵢyᵢ))    products held at true magnitude

Both fall back to IEEE propagation on a non-finite addend or a genuine overflow
past the accumulator's range (finite inputs never yield NaN). There is no SIMD
kernel — the integer chunk sweep does not vectorize — so under `-d:simd` the C
ABI dispatches to the scalar algorithm (ADR-0006).
"""

nbCode:
  let bigCancel = [1.0, 1e100, 1.0, -1e100]
  echo "superSum (magnitude) = ", superSum(bigCancel)
  echo "shewchukSum         = ", shewchukSum(bigCancel)
  let dotX = [1.0, 2.0, 3.0]
  let dotY = [4.0, 5.0, 6.0]
  echo "superDot             = ", superDot(dotX, dotY)

nbText: """
The C ABI exposes `ua_sum_exact` / `ua_dot_exact` and Python exposes
`exact_sum` / `exact_dot`.

### References

- Neal, R. (2015). "Fast Exact Summation Using Small and Large
  Superaccumulators". arXiv:1505.05571 — the small superaccumulator layout,
  lazy carry, and 64×64→128 product accumulation.
"""

nbText: """
## Faithful and correctly-rounded summation (ORO / Rump)

The compensated and exact sums above split the work per *element*; the
Ogita–Rump–Oishi and Rump families split it per *order of rounding error*. Both
build on the error-free `transform` (ORO Vecsum, Alg 4.6) rather than per-step
feedback, so the error vector is distilled once and re-summed.

`sum2` is ORO Alg 4.1 — the magnitude-robust compensated sum, value-identical to
`neumaierSum` (the ORO Vecsum-with-final-correction *is* Neumaier's scheme),
exposed under the ORO name for API symmetry. `sumK` is ORO Alg 4.8: the
`transform` applied K times, summing the partials, so each extra cascade level
compensates one order of rounding error — K=1 the naive `twoSum` chain, K=2 the
first-order compensated sum (≈ `neumaierSum`), K=3 the second-order (≈
`kleinSum`).

    sum2:  (2u + O(nu²)) · Σ|xᵢ|                    ≈ neumaierSum (ORO Alg 4.1)
    sumK:  γ_{n-1}^K · Σ|xᵢ| / (1 - γ_{n-1}^K)      (ORO Alg 4.8)

The bound is in `E = Σ|xᵢ|`, not `|S|`: cancellation grows the *relative* error,
not the absolute. For a result in `|S|` (correctly rounded) use `shewchukSum`
or `superSum`; for one between the two — *faithful*, no float lies between the
result and the exact sum — use `accSum` (Rump Alg 4.5), and for the exact
round-to-nearest use `nearSum` (Rump Alg 7.4):

    accSum:  < 2·eps·ufp(result)        faithful, 1 ulp (Rump Alg 4.5)
    nearSum: ½ ulp(fl(Σ xᵢ))            correctly rounded (Rump Alg 7.4)

`accSum` runs Rump's `Transform` repeat-until over `ExtractVector`; runtime grows
with `log₂(cond(sum))`. `nearSum` runs it once for a faithful result and the
residual, then resolves the rounding direction with a second `TransformK(p',
R − δ')` offset to the midpoint between the two candidate floats (Lemma 7.3):
its sign picks predecessor / result / successor, and the zero case is the exact
tie. The correctly-rounded guarantee holds under `2^(2M)·eps ≤ 1` (`M =
⌈log₂(n+2)⌉`): `n ≤ 6.7e7` (float64), `n ≤ 4094` (float32).

`conditionNumber` reports `cond(sum) = Σ|xᵢ| / |Σxᵢ|` (Higham 2002, §4.1): `1`
for a no-cancellation sum, large under catastrophic cancellation, `+inf` when the
sum is exactly `0` or `Σ|xᵢ|` overflows, `0` for empty input — the quantity the
ORO/Rump bounds are stated in.

Non-finite input or a `sigma0` overflow (input near the float max, where the sum
itself overflows) falls back to `superSum`, which is correctly rounded and so
still faithful — finite inputs never yield NaN. There is no SIMD kernel — the
sequential cascade does not vectorize — so under `-d:simd` the C ABI dispatches
to the scalar algorithm (ADR-0006).
"""

nbCode:
  let tenths = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  echo "sum2   (0.1x10) = ", sum2(tenths)
  echo "sumK K=2         = ", sumK(tenths, 2)
  echo "accSum           = ", accSum(tenths)
  echo "nearSum          = ", nearSum(tenths)
  echo "cond (magnitude) = ", conditionNumber([1.0, 1e100, 1.0, -1e100])

nbText: """
"""

nbSave
