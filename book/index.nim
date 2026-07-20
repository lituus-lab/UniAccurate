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
