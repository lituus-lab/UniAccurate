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
