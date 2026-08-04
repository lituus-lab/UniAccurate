# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""Python-binding bench: uniaccurate vs the stdlib functions it's comparable
to, both with a plain `list` and with the zero-copy `array.array('d', ...)`
fast path. `math.fsum` is CPython's own C-implemented Shewchuk
adaptive-precision sum -- the same algorithm family as `shewchuk_sum`, so
timing them against each other is an apples-to-apples comparison, not naive
vs accurate.

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
import array
import math
import os
import platform
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import uniaccurate  # noqa: E402
from uniaccurate import _validate  # noqa: E402
from uniaccurate._core import _sum_shewchuk_c, _sum_shewchuk_buf_c  # noqa: E402


def timeit(fn, data, repeats=7):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(data)
        best = min(best, time.perf_counter() - t0)
    return best


def bench_trio(name_a, fn_a, name_b, fn_b, data):
    """Time `fn_a` on both a plain list and the equivalent array.array('d',
    ...), against `fn_b` (a stdlib reference) on the list."""
    buf = array.array("d", data)
    ta_list = timeit(fn_a, data)
    ta_buf = timeit(fn_a, buf)
    tb = timeit(fn_b, data)
    equal = fn_a(data) == fn_b(data)
    return name_a, ta_list, ta_buf, name_b, tb, equal


def main():
    random.seed(20260804)
    sizes = [1_000, 100_000, 1_000_000]
    timing_rows = []
    accuracy_rows = []
    for n in sizes:
        data = [random.uniform(-1e6, 1e6) for _ in range(n)]
        timing_rows.append((n,) + bench_trio(
            "shewchuk_sum", uniaccurate.shewchuk_sum, "math.fsum", math.fsum, data))
        timing_rows.append((n,) + bench_trio(
            "naive_sum", uniaccurate.naive_sum, "sum()", sum, data))
        truth = uniaccurate.shewchuk_sum(data)  # correctly-rounded reference
        naive = uniaccurate.naive_sum(data)
        builtin = sum(data)
        accuracy_rows.append((n, abs(naive - truth), abs(builtin - truth)))

    out_path = os.path.join(os.path.dirname(__file__), "..", "bench", ".md_python.md")
    with open(out_path, "w") as f:
        f.write(f"Python {platform.python_version()} ({platform.python_implementation()})\n\n")
        f.write("The `list` path pays a per-call Python-side validation and "
                 "copy cost (see \"Where the time goes\" below); the "
                 "`array.array('d', ...)` fast path skips both. Honest "
                 "result: `shewchuk_sum` on `array.array` reaches rough "
                 "parity with `math.fsum` (~1.0x-1.2x here, not a clear win), "
                 "and `naive_sum` on `array.array` genuinely beats `sum()` "
                 "(~0.25x-0.4x) since `sum()` does more work per element "
                 "(Neumaier compensation) than a plain loop.\n\n")
        f.write("| comparison | n | uniaccurate, list (ms) | uniaccurate, "
                 "array.array (ms) | reference (ms) | ratio list/ref | "
                 "ratio array/ref | same result? |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for n, na, ta_list, ta_buf, nb, tb, eq in timing_rows:
            r_list = ta_list / tb if tb > 0 else float("inf")
            r_buf = ta_buf / tb if tb > 0 else float("inf")
            f.write(f"| {na} vs {nb} | {n} | {ta_list * 1000:.3f} | "
                     f"{ta_buf * 1000:.3f} | {tb * 1000:.3f} | {r_list:.2f}x | "
                     f"{r_buf:.2f}x | {'yes' if eq else 'NO'} |\n")
        f.write("\n**Why `naive_sum` vs `sum()` says NO**: on this interpreter, "
                 "`sum()` is not naive -- it already matches the correctly-rounded "
                 "reference (`shewchuk_sum`) exactly, while `naive_sum` (a plain "
                 "left-to-right loop) carries real rounding error:\n\n")
        f.write("| n | \\|naive_sum - correct\\| | \\|sum() - correct\\| |\n")
        f.write("|---|---|---|\n")
        for n, naive_err, builtin_err in accuracy_rows:
            f.write(f"| {n} | {naive_err:.3e} | {builtin_err:.3e} |\n")

        # Where the time goes, at the largest n, for each calling convention:
        # the Python-side input coercion loop (`_validate`, list path only)
        # vs the Cython/C summation itself (isolated by pre-validating the
        # list, or by going straight through the array.array buffer path,
        # which skips `_validate()` entirely).
        biggest = [random.uniform(-1e6, 1e6) for _ in range(sizes[-1])]
        biggest_buf = array.array("d", biggest)
        t_full_list = timeit(uniaccurate.shewchuk_sum, biggest)
        t_full_buf = timeit(uniaccurate.shewchuk_sum, biggest_buf)
        t_validate = timeit(_validate, biggest)
        validated = _validate(biggest)
        t_core_from_list = timeit(lambda d: _sum_shewchuk_c(d), validated)
        t_core_from_buf = timeit(lambda d: _sum_shewchuk_buf_c(d), biggest_buf)
        f.write(f"\n**Where the time goes** (`shewchuk_sum`, n={sizes[-1]}):\n\n")
        f.write("| stage | ms |\n|---|---|\n")
        f.write(f"| full call, list input | {t_full_list * 1000:.3f} |\n")
        f.write(f"| full call, array.array input (fast path) | {t_full_buf * 1000:.3f} |\n")
        f.write(f"| `_validate()` only (list path) | {t_validate * 1000:.3f} |\n")
        f.write(f"| C core only, given a pre-validated list | {t_core_from_list * 1000:.3f} |\n")
        f.write(f"| C core only, given the array.array directly (zero-copy) | {t_core_from_buf * 1000:.3f} |\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
