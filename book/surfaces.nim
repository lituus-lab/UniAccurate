# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniAccurate

nbInit(theme = useNimibook)
useLituus()
nb.title = "Surfaces"

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
uniaccurate.version()            # '1.1.0'
```

Here the input check is expressed as an exception, because Python has
exceptions to carry it: a non-numeric argument raises `TypeError`. Each
surface expresses one contract in the terms its own callers expect — a
precondition in Nim, a defined fallback in C, an exception in Python.

`py/notebooks/quickstart.ipynb` runs these calls against an installed wheel
and renders on GitHub directly.
"""

nbSave
