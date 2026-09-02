# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""Author py/notebooks/quickstart.ipynb, then execute it so the committed file
carries real outputs for GitHub to render. Run from the repo root:

    python3 py/notebooks/build_quickstart.py

CI re-executes the notebook against an installed wheel; this script only
regenerates it after an API change."""
import os

import nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "quickstart.ipynb")

CELLS = [
    ("md", """# UniAccurate — Python quickstart

`uniaccurate` is a Cython extension over the UniAccurate C ABI, shipped as a
self-contained wheel: the native library travels inside the package, so
installing it needs neither Nim nor a compiler.

```
pip install lituus-uniaccurate
```

CI executes this notebook against the wheel the release actually publishes, so
an output below that stops matching fails the build."""),
    ("md", "## The API"),
    ("code", """import uniaccurate

uniaccurate.version(), uniaccurate.__version__"""),
    ("md", """`two_sum` is an error-free transformation: it returns the rounded
sum `s = fl(a + b)` and the exact rounding error `e`, with `a + b == s + e` in
real arithmetic."""),
    ("code", "uniaccurate.two_sum(1.0, 2.0)"),
    ("md", """When the addend is too small to affect the sum, it is not lost —
it shows up as `e`. That recovered error is what compensated summation
threads through a long running sum."""),
    ("code", "uniaccurate.two_sum(1.0, 2e16)"),
    ("md", """Non-finite input is not a contract violation: `s` follows IEEE
arithmetic and the error reads `NaN`."""),
    ("code", """import math
s, e = uniaccurate.two_sum(float("inf"), 1.0)
math.isinf(s), math.isnan(e)"""),
    ("md", "A non-numeric argument is a type error, not a coercion."),
    ("code", """try:
    uniaccurate.two_sum("x", 2.0)
except TypeError as exc:
    print("TypeError:", exc)"""),
    ("md", """## The C ABI underneath

The same entry point is reachable from anything that speaks C. There the
contract is expressed without raising — an exception must never unwind across
an ABI boundary:

```c
double s, e;
ua_two_sum(1.0, 2e16, &s, &e);   /* s = 2e16, e = 1.0 */
```

See `include/UniAccurate.h`, and the book for the full picture."""),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    # Execute from the repo root, never from py/: there, `import uniaccurate`
    # would resolve to the py/uniaccurate source tree instead of the installed
    # package, and the notebook would stop testing what it claims to test.
    NotebookClient(nb, timeout=120, kernel_name="python3",
                   resources={"metadata": {"path": ROOT}}).execute()
    with open(OUT, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
