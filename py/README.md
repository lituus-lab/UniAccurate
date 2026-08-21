<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# uniaccurate

Error-free transforms and accurate summation/dot-product for floating-point
arithmetic — a Python binding over [UniAccurate](https://github.com/lituus-lab/UniAccurate),
a Nim library with a C ABI core.

Naive summation accumulates `O(n)` rounding error. `uniaccurate` gives you
progressively stronger alternatives without leaving Python: pairwise
(`O(log n)`), compensated (Kahan/Neumaier/Klein), and correctly-rounded
(Shewchuk's adaptive-precision expansion, the Neal small superaccumulator) —
plus the matching compensated and correctly-rounded dot products.

## Install

```bash
pip install uniaccurate
```

Prebuilt wheels ship for Linux, macOS and Windows, CPython 3.9-3.13. No Nim
or C toolchain needed at install time.

## Quick start

```python
import uniaccurate

s, e = uniaccurate.two_sum(1.0, 2e16)                # (2e16, 1.0) -- error recovered exactly
uniaccurate.kahan_sum([0.1] * 10)                    # 1.0
uniaccurate.shewchuk_sum([1.0, 1e100, 1.0, -1e100])  # 2.0, correctly rounded
uniaccurate.exact_dot([1.0, 2.0], [3.0, 4.0])        # 11.0, correctly rounded
uniaccurate.scaled_norm([3.0, 4.0])                   # 5.0
```

## What's included

| Category | Functions |
|---|---|
| Error-free transform | `two_sum` |
| Summation | `naive_sum`, `pairwise_sum`, `pairwise_sum_iterative` |
| Compensated summation | `kahan_sum`, `neumaier_sum`, `klein_sum`, `sum_k` |
| Correctly-rounded summation | `shewchuk_sum`, `exact_sum`, `oro_sum`, `acc_sum`, `near_sum` |
| Dot product | `naive_dot`, `dot2`, `dot_k`, `exact_dot` |
| Diagnostics | `condition_number` |
| Statistical reductions | `scaled_mean`, `scaled_norm`, `centered_sum_squares`, `centered_cross_product`, `centered_cosine_similarity` |

Every function raises `TypeError` on non-numeric input rather than silently
coercing; see each docstring (`help(uniaccurate.kahan_sum)`) for the exact
error/NaN/Inf behavior.

## Links

- Source, Nim API, C ABI, ADRs: <https://github.com/lituus-lab/UniAccurate>
- Issues: <https://github.com/lituus-lab/UniAccurate/issues>
- License: Apache-2.0 (`LICENSE` in the source repo)

## Development

Building from source (contributing, or a platform without a prebuilt wheel)
needs a Nim toolchain — see `CONTRIBUTING.md` in the source repo.

```bash
nimble pyLib                                             # build the shared lib the extension links against
cd py && python3 setup.py build_ext --inplace            # build extension
cd py && python3 -m pytest -q                            # test
```
