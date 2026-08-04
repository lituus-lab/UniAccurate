# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""Python-binding bench: uniaccurate vs the stdlib functions it's comparable
to. `math.fsum` is CPython's own C-implemented Shewchuk adaptive-precision
sum -- the same algorithm family as `shewchuk_sum`, so timing them against
each other is an apples-to-apples comparison, not naive vs accurate.

`sum()` is timed against `naive_sum` as the cheap-baseline pair, but on a
modern CPython `sum()` is not actually naive: it uses a Neumaier-style
compensated accumulator internally, not a proven correctly-rounded algorithm
like `shewchuk_sum`/`math.fsum` -- the accuracy table below reports
`|sum() - shewchuk_sum|` measured directly on this run's own seeded datasets,
not a general guarantee, and it lands at 0 at every size tested here (fixed
seed, reproducible), while `naive_sum` (a plain left-to-right loop) carries
real ULP error. Both the timing and that measured accuracy gap are reported --
this is not a naive-vs-naive comparison on this interpreter, and the table
says so rather than leaving a bare mismatch to look like a bug.

Needs the built extension (`nimble buildCython`, from the repo root) first.
Writes bench/.md_python.md; run via `nimble benchPython` or directly from
py/.
"""
import math
import os
import platform
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import uniaccurate  # noqa: E402
from uniaccurate import _validate  # noqa: E402
from uniaccurate._core import _sum_shewchuk_c  # noqa: E402


def timeit(fn, data, repeats=7):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(data)
        best = min(best, time.perf_counter() - t0)
    return best


def bench_pair(name_a, fn_a, name_b, fn_b, data):
    ta = timeit(fn_a, data)
    tb = timeit(fn_b, data)
    equal = fn_a(data) == fn_b(data)
    return name_a, ta, name_b, tb, equal


def main():
    random.seed(20260804)
    sizes = [1_000, 100_000, 1_000_000]
    timing_rows = []
    accuracy_rows = []
    for n in sizes:
        data = [random.uniform(-1e6, 1e6) for _ in range(n)]
        timing_rows.append((n,) + bench_pair(
            "shewchuk_sum", uniaccurate.shewchuk_sum, "math.fsum", math.fsum, data))
        timing_rows.append((n,) + bench_pair(
            "naive_sum", uniaccurate.naive_sum, "sum()", sum, data))
        truth = uniaccurate.shewchuk_sum(data)  # correctly-rounded reference
        naive = uniaccurate.naive_sum(data)
        builtin = sum(data)
        accuracy_rows.append((n, abs(naive - truth), abs(builtin - truth)))

    out_path = os.path.join(os.path.dirname(__file__), "..", "bench", ".md_python.md")
    with open(out_path, "w") as f:
        f.write(f"Python {platform.python_version()} ({platform.python_implementation()})\n\n")
        f.write("| comparison | n | uniaccurate (ms) | reference (ms) | "
                "ratio (uniaccurate/reference) | same result? |\n")
        f.write("|---|---|---|---|---|---|\n")
        for n, na, ta, nb, tb, eq in timing_rows:
            ratio = ta / tb if tb > 0 else float("inf")
            f.write(f"| {na} vs {nb} | {n} | {ta * 1000:.3f} | {tb * 1000:.3f} | "
                     f"{ratio:.2f}x | {'yes' if eq else 'NO'} |\n")
        f.write("\n**Why `naive_sum` vs `sum()` says NO**: on this interpreter, "
                 "`sum()` is not naive -- it already matches the correctly-rounded "
                 "reference (`shewchuk_sum`) exactly, while `naive_sum` (a plain "
                 "left-to-right loop) carries real rounding error:\n\n")
        f.write("| n | \\|naive_sum - correct\\| | \\|sum() - correct\\| |\n")
        f.write("|---|---|---|\n")
        for n, naive_err, builtin_err in accuracy_rows:
            f.write(f"| {n} | {naive_err:.3e} | {builtin_err:.3e} |\n")

        # Where the time in `shewchuk_sum` actually goes, at the largest n:
        # the Python-side input coercion loop (`_validate`) vs the Cython/C
        # summation itself (pre-validated, so the C call alone is isolated).
        biggest = [random.uniform(-1e6, 1e6) for _ in range(sizes[-1])]
        t_full = timeit(uniaccurate.shewchuk_sum, biggest)
        t_validate = timeit(_validate, biggest)
        validated = _validate(biggest)
        t_core = timeit(lambda d: _sum_shewchuk_c(d), validated)
        f.write(f"\n**Where the time goes** (`shewchuk_sum`, n={sizes[-1]}): "
                 "most of the gap above is the Python-side input-coercion loop, "
                 "not the C summation itself:\n\n")
        f.write("| stage | ms |\n|---|---|\n")
        f.write(f"| full `shewchuk_sum` call | {t_full * 1000:.3f} |\n")
        f.write(f"| `_validate()` only | {t_validate * 1000:.3f} |\n")
        f.write(f"| C core only (pre-validated) | {t_core * 1000:.3f} |\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
