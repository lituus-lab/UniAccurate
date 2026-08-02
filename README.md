<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# UniAccurate

arithmetic: naive, pairwise, and compensated sums, and compensated dot
products, in Nim, with a hand-written C ABI and a Cython Python binding.

Correctly-rounded alternatives (Shewchuk's adaptive-precision expansion, the
Neal small superaccumulator) are also available. Layer-1 in the `lituus-lab`
`Uni*` family DAG: depends only on `nimsimd` (optional, `-d:simd`).

## Quick start

```nim
import UniAccurate

let (s, e) = twoSum(1.0, 2e16)          # (2e16, 1.0) -- error recovered exactly
echo kahanSum(@[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 1.0
echo shewchukSum(@[1.0, 1e100, 1.0, -1e100])                        # 2.0, correctly rounded
```

```c
#include "UniAccurate.h"
double s, e;
ua_two_sum(1.0, 2e16, &s, &e);          // s = 2e16, e = 1.0
```

```python
import uniaccurate
uniaccurate.two_sum(1.0, 2e16)          # (2e16, 1.0)
```

See `book/index.nim` (nimib, built into `book/index.html`) for the full walkthrough
with error bounds and references, and `py/notebooks/quickstart.ipynb` for the
Python side.

## Layout

```
src/UniAccurate.nim          umbrella module
src/UniAccurate/twosum.nim   error-free transforms (NimContracts)
src/UniAccurate/algorithms/  summation and dot-product kernels (NimContracts)
src/UniAccurate/simd.nim     SIMD kernels, gated by `-d:simd` (AVX-512/AVX2/NEON)
src/UniAccurate/c_api.nim    C ABI
include/UniAccurate.h        hand-written C header
tests/test_*.nim             Nim tests (eft, sums, compensated, shewchuk, exact, oro, dot, property, simd)
tests/c/                     C ABI test (links the header against the lib)
examples/                    Nim + C demos
py/                          Cython binding + pytest
bench/                       cross-backend perf/accuracy comparator vs the C originals and Rust `accurate` (ADR-0007)
ADRs/                        0001 optional deps, 0002 license, 0003 C ABI & Python, 0004 FMA, 0005 SIMD, 0006 new algos, 0007 perf levers
.github/workflows/ci.yml     3-OS Nim matrix + C ABI + Python + SIMD + bench smoke
```

## Build

```bash
nimble install -y
nimble test           # Nim, debug (contracts active)
nimble testRelease    # Nim, release (contracts compiled away)
nimble testAll        # debug + release + C ABI
nimble simd           # SIMD tests, -d:simd (AVX2 amd64, NEON arm64)
nimble ctest          # C ABI: static lib + tests/c
nimble cexample       # C demo
nimble example        # Nim demo
nimble pyTest         # Cython + pytest
nimble bench          # quick scalar bench smoke -> bench/compare/*.csv
nimble coverage       # gcov + lcov -> coverage/
nimble book           # nimib book -> book/index.html
nimble docs           # book + API reference -> pages/
```

## CI

`test`, `cabi` and `python` on ubuntu/macOS/Windows. `consume-cabi` and
`consume-wheel` rebuild against the published artifacts on a machine without Nim,
so what ships is what was tested. `bench` runs a reduced-size smoke on
ubuntu/macOS — correctness/non-regression only, not the AVX-512 reference
numbers from ADR-0007 (GitHub runners are not guaranteed AVX-512 hardware;
that comparison stays a manual FreeBSD/Zen4 check). `coverage` and `docs` run
on ubuntu.

`dco` blocks PRs missing a `Signed-off-by` trailer; `commitizen` blocks PRs whose
commits or title are not [Conventional Commits](https://www.conventionalcommits.org/)
(`CONTRIBUTING.md`).

The same gates run locally with pre-commit: `pip install pre-commit && pre-commit install`
(`CONTRIBUTING.md`).

`docs` publishes to GitHub Pages — skipped on push to a fork or while the repo
is private, on by default once public on `main`.

## AI-assisted contributions

Assistance from AI/LLM tools is welcome on the same terms as any other
contribution.

- **Accountability.** The human contributor is the author and remains fully
  responsible for the change. The DCO sign-off (`Signed-off-by`) is the mechanism:
  by signing you certify the content is yours or properly licensed — this covers
  AI-assisted work, provided you can stand behind it.
- **No third-party contamination.** Ensure AI output introduces no code from a
  third party without a compatible license and attribution. If an LLM reproduced
  protected material, do not submit it.
- **Correctness is yours.** The gates (tests, `nimble lint`, conventional commits,
  pre-commit) catch a lot, but you own the result — review and verify what you
  commit.
- **Atomic commits.** Each commit is one logical change. A PR may stack
  several atomic commits (one per element, say) — one monolithic big-bang
  commit is not.
- **Disclosure.** State in the PR whether AI assistance was used (see the PR
  template). It is not a hard requirement — the DCO remains the gate.

## License

Apache-2.0 (`LICENSE`). DCO sign-off on every commit (`CONTRIBUTING.md`).
