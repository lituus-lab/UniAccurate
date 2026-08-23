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

## What's inside

- **Error-free transforms** (`twosum.nim`) — `twoSum`/`twoSumFast`, `twoProduct`
  (and its FMA form): split a float sum/product into `(result, exactError)`
  with no rounding lost, the building block everything below is composed from.
- **Summation** (`algorithms/`) — naive, pairwise (tree-reduction), compensated
  (Kahan, Neumaier, Klein), Shewchuk's adaptive-precision expansions, an exact
  superaccumulator (Neal), and the ORO/Rump family (`sum2`/`sumK`,
  `nearSum`/`accSum`).
- **Dot products** (`algorithms/dotproduct.nim`) — naive, compensated
  (`dot2`/`dotK3`), and a correctly-rounded `superDot`.
- **Statistical reductions** (`algorithms/statistical_reductions.nim`) —
  range-safe mean and Euclidean norm, exact sums of represented centered
  products, centered cosine similarity, centered projection coefficients, and
  reusable scale-separated centered norm states
  without raw square overflow.
- **SIMD kernels** (`simd.nim`) — AVX2/AVX-512 (amd64) and NEON (arm64)
  vectorized sum/dot, see ADR-0005 and ADR-0007. The direct Nim API and the
  sum family stay opt-in via `-d:simd`; the C ABI dot family (`ua_dot_naive`/
  `ua_dot2`/`ua_dot_k`, including the Python wheels built on it) picks
  AVX2/AVX-512 at runtime via CPUID unconditionally on amd64, no `-d:simd`
  needed, see ADR-0008.

## The Uni* family

UniAccurate is layer 1 of `lituus-lab`'s `Uni*` family: a set of Nim libraries,
each with a C ABI and a Python binding, unified by a shared dependency DAG and
documentation/testing conventions. See
[lituus-lab/.github](https://github.com/lituus-lab/.github) for the family's
purpose and philosophy. UniAccurate itself depends on nothing else in the
family (only the optional `nimsimd`); it exists to give downstream libraries —
starting with UniMath's `BigFloat` rounding and UniLinalg's iterative
refinement — an exact-accumulation primitive to build on.

## Provenance & development

The summation/dot algorithms here are textbook (Kahan, Neumaier, Klein,
Shewchuk, Ogita-Rump-Oishi, Neal's superaccumulator) — no original numerics,
gathered from the papers cited in `book/index.nim` and cross-checked against
the C originals and the Rust `accurate` crate in `bench/`. The Nim
implementation descends from an earlier hand-written `AccurateSums` (see
`.github/README.md`).

Development used LLM/agent assistance extensively, on the terms described
below. One visible consequence: this repo's git history is short and linear,
with commits landing close together in time — that reflects an LLM/agent
rewrite pass over a pre-existing design, not the numerics being designed at
that speed from a blank page.

## Layout

```text
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
nimble benchAll       # full cross-backend comparator -> bench/compare/summary.md (manual, see Benchmarks)
nimble benchReadme    # benchAll, then splice a headline table into this README for this machine
nimble coverage       # gcov + lcov -> coverage/
nimble book           # nimib book -> book/index.html
nimble docs           # book + API reference -> pages/
```

## Benchmarks

`nimble bench` is a reduced-size CI smoke test (correctness/non-regression
only). For real numbers: run `benchData`, then one `bench<Backend>` task per
column (`benchNim`, `benchNimFma`, `benchNimSimd`, `benchC`, `benchCFma`,
`benchRust`/`benchRustFma` — the last two need `cargo`), then `benchAll` to
time every algorithm against the C originals and the Rust `accurate` crate on
the same canonical datasets and join all backends against one
correctly-rounded oracle. Manual only, never in CI.

`nimble benchReadme` runs the same aggregation and additionally writes a
headline table below, tagged to the machine it ran on
(`<!-- bench:machine=... -->` — see `bench/aggregate.nim`'s `spliceReadme`).
Re-running on the same machine replaces only that machine's block; a second
machine (say a FreeBSD/Zen4 box, `UNIACCURATE_BENCH_MACHINE` env var to name
it explicitly) adds its own block alongside, so this table can carry more
than one machine's numbers at once without either overwriting the other.

<!-- markdownlint-disable MD013 MD036 -->
<!-- bench:insert -->

<!-- bench:machine=macosx-apple-m4 -->
All times ns/elem (f64, largest common n) -- lower is faster. `nim/c` is UniAccurate's default column against the honest external C reference (below 1.0 would mean UniAccurate is faster). Full matrix (all algorithms, f32 included): `nimble benchAll` locally, see `bench/compare/summary.md` (generated, not tracked).

| algo | n | nim (ns/elem) | nim_fma (ns/elem) | nim_simd (ns/elem) | nim_fmadot (ns/elem) | c (ns/elem) | c_fma (ns/elem) | rust (ns/elem) | rust_fma (ns/elem) | nim/c |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 1000000 | 0.5326 | 0.5497 | 0.5387 | 0.5416 | 0.5378 | - | 0.5428 | 0.5396 | 0.99x |
| pairwise | 1000000 | 0.2129 | 0.2222 | 0.2126 | 0.2116 | 0.3169 | - | - | - | 0.67x |
| kahan | 1000000 | 2.2325 | 2.2226 | 2.2035 | 2.1970 | 2.1295 | - | 2.1524 | 2.1449 | 1.05x |
| dot2 | 1000000 | 2.7865 | 2.9057 | 2.8034 | 2.7659 | 0.7327 | - | 1.5882 | 1.3764 | 3.80x |
| shewchuk | 1000000 | 12.7226 | 13.4481 | 12.7184 | - | 12.2438 | - | - | - | 1.04x |

**Python binding vs stdlib** (`nimble benchPython`)

Python 3.14.6 (CPython)

| comparison | n | uniaccurate, list (ms) | uniaccurate, array.array (ms) | reference (ms) | ratio list/ref | ratio array/ref | same result? |
|---|---|---|---|---|---|---|---|
| shewchuk_sum vs math.fsum | 1000 | 0.011 | 0.007 | 0.007 | 1.71x | 1.07x | yes |
| naive_sum vs sum() | 1000 | 0.008 | 0.001 | 0.003 | 2.56x | 0.38x | NO |
| shewchuk_sum vs math.fsum | 100000 | 1.334 | 0.947 | 0.865 | 1.54x | 1.09x | yes |
| naive_sum vs sum() | 100000 | 0.467 | 0.055 | 0.183 | 2.55x | 0.30x | NO |
| shewchuk_sum vs math.fsum | 1000000 | 12.780 | 8.879 | 8.780 | 1.46x | 1.01x | yes |
| naive_sum vs sum() | 1000000 | 4.409 | 0.479 | 1.902 | 2.32x | 0.25x | NO |

| n | \|naive_sum - correct\| | \|sum() - correct\| |
|---|---|---|
| 1000 | 9.313e-09 | 0.000e+00 |
| 100000 | 7.749e-07 | 0.000e+00 |
| 1000000 | 1.568e-05 | 0.000e+00 |

| shewchuk_sum call, n=1000000 | ms |
|---|---|
| list input | 12.974 |
| array.array input | 9.086 |

<!-- /bench:machine=macosx-apple-m4 -->

<!-- markdownlint-enable MD013 MD036 -->

See ADR-0007 for the full lever-by-lever analysis (the `isFin` bit-mask, FMA
enablement, `assumeFinite`, `useFastTwoSum`) and `bench/compare/summary.md`
(generated locally, not tracked) for every algorithm and both `f32`/`f64`.

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
