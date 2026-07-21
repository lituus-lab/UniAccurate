# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib

nbInit
nb.title = "UniAccurate"

nbText: """
# UniAccurate

Error-free transformations (EFT) for floating-point arithmetic, exposed across
three surfaces: **Nim**, a **C ABI**, and a **Python** binding.

An EFT computes an operation — here, addition — and returns both the rounded
result `s = fl(a + b)` and the exact rounding error `e`, such that in real
arithmetic `a + b = s + e` exactly. The error is not lost; compensated
summation threads it back through a running sum. This page is a nimib book:
every Nim block below is compiled and run when the book is built, and the
output shown is what the code actually produced. A change that breaks the API
breaks the docs build, so the two cannot drift apart.

## The Nim surface

The umbrella module re-exports every public submodule.
"""

nbCode:
  import UniAccurate

  echo "version ", UniAccurateVersion
  let (s, e) = twoSum(1.0, 2.0)
  echo "twoSum(1.0, 2.0) = (", s, ", ", e, ")"
  let (s2, e2) = twoSum(1.0, 2e16)
  echo "twoSum(1.0, 2e16) = (", s2, ", ", e2, ")"

nbText: """
`2e16 + 1` rounds back to `2e16` in float64 — the `1` is below the ULP, so a
plain `+` drops it. `twoSum` recovers it as `e = 1.0`: the identity `a + b =
s + e` still holds to the last bit.

## The contract is part of the signature

`twoSum` is the Møller–Knuth form: six FLOPs, branchless, no precondition on
operand ordering. Its postcondition states the non-overlap bound `|e| <=
½ ulp(s)` for normal `s` — a property cheaper to check than the body is to
run, and never re-derived by calling the function again.

The contract is written with NimContracts (`require:` / `ensure:` / `body:`).
Under `-d:release` it compiles away entirely: the release build pays nothing,
while debug builds and the test suite catch a violation at the call site.
"""

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
algorithm (documented in ADR-0007).

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
ABI dispatches to the scalar algorithm (ADR-0007).
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
not the absolute. For a result in `|S|` (correctly rounded) use `shewchuckSum`
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
to the scalar algorithm (ADR-0007).
"""

nbCode:
  let tenths = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  echo "sum2   (0.1x10) = ", sum2(tenths)
  echo "sumK K=2         = ", sumK(tenths, 2)
  echo "accSum           = ", accSum(tenths)
  echo "nearSum          = ", nearSum(tenths)
  echo "cond (magnitude) = ", conditionNumber([1.0, 1e100, 1.0, -1e100])

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
## The C ABI

The same entry point, reachable from anything that speaks C. The header is
hand-written and kept in sync with `src/UniAccurate/c_api.nim`; `tests/c` links
one against the other on every CI run, so a drift is caught rather than
shipped.

```c
const char *ua_version(void);
void ua_two_sum(double a, double b, double *s, double *e);
```

The C ABI **never raises**. For non-finite input, `*s` follows IEEE
arithmetic and `*e` reads `NaN` — an exception must never unwind across the
ABI boundary, which would be undefined behaviour.

## The Python surface

A Cython extension over the C ABI, shipped as a self-contained wheel: the
library travels inside the package, so installing it needs neither Nim nor a
compiler.

```python
import uniaccurate

uniaccurate.two_sum(1.0, 2e16)   # (2e16, 1.0) — error recovered
uniaccurate.version()            # '0.1.0'
```

Here the input check is expressed as an exception, because Python has
exceptions to carry it: a non-numeric argument raises `TypeError`. Each
surface expresses one contract in the terms its own callers expect — a
precondition in Nim, a defined fallback in C, an exception in Python.

`py/notebooks/quickstart.ipynb` runs these calls against an installed wheel
and renders on GitHub directly.

## References

The EFT identities, their exactness conditions, and the FMA contraction
invariant are documented in `src/UniAccurate/twosum.nim` with full citations
(Dekker 1971; Møller 1965; Knuth 1998; Ogita, Rump & Oishi 2005; Boldo &
Melquiond 2008; Shewchuk 1997; Goldberg 1991). The generated API reference
lists the symbols; this book is where the layer gets explained.
"""

nbSave
