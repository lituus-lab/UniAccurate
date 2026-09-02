<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# AGENTS.md — UniAccurate

## Build & gates

```bash
nimble install -y
nimble testAll    # Nim debug + release + C ABI
nimble simd       # SIMD tests, -d:simd (AVX2 amd64, NEON arm64)
nimble pyTest     # Cython + pytest (needs libUniAccurate.so)
nimble example
nimble coverage   # gcov + lcov -> coverage/ (needs lcov; linux/macOS)
nimble docs       # nimib book + API reference -> pages/ (needs nimib)
```

`nimble docs` needs a complete Nim distribution: `--project` builds `dochack`,
which Homebrew's `nim` omits (no `tools/`). choosenim and the CI action ship it.

CI: 3-OS Nim matrix + C ABI + Python (ubuntu/macOS/Windows) + SIMD (ubuntu AVX2, macOS NEON).

## Testing

Three tiers, each catching what the others cannot:

- **Structural tests** (`tests/test_*.nim`, `nimble test`): debug builds with
  contracts active. `test_eft`, `test_naivesum`, `test_pairwisesum`,
  `test_compensatedsum`, `test_shewchuksum`, `test_exactsum`, `test_orosum`,
  `test_dotproduct` lock exact
  behavior — empty/single inputs, the compensation win on `0.1·10`,
  magnitude-robustness, non-finite propagation, and the `s + e == a op b`
  identity to the last bit.
- **Randomized property tests** (`tests/test_property.nim`, `nimble prop`):
  oracle-free invariants at scale on the native path — finite inputs never
  yield NaN (bounded and overflow-prone, float64 and float32), and every sum
  agrees exactly on integer data within 2^53. Deterministic xorshift64 so a
  failure reproduces with no RNG state.
- **Exact-rational oracle** (`py/tests/test_oracle.py`, `nimble pyTest`):
  forward-error-bound checks through the C ABI. `fractions.Fraction` gives the
  exact real sum `S` and magnitude `E`; the bound is computed in the same exact
  arithmetic, so a failure is a real bound violation, not float noise. Catches
  a regression the float-level tests cannot (an algorithm that stays finite but
  drifts past its published error bound).

## Conventions

- English comments, terse, describe what is done. No "deprecated".
- NimContracts `{.contractual.}` + `require:`/`ensure:`/`body:`, compiled away
  under `-d:release`. C ABI never raises — it clamps out-of-range input.
- A postcondition is cheaper than the body: never re-derives the result by
  calling the function itself.
- C ABI: hand-written `include/UniAccurate.h` kept in sync with
  `src/UniAccurate/c_api.nim`; `tests/c` links the header against the lib.
  Built `--app:staticlib`/`--app:lib --noMain --mm:arc -d:release`.
- A change to `c_api.nim` is verified by `ctest`, `pyTest` and, where there
  is one, `wasmTest`: three linkages, three runtime bootstraps. A green
  `ctest` alone proved nothing the day the shared build lost its
  initializer and every registry answered with the sentinel.
- C symbols `ua_*`; lib `libUniAccurate`; header `UniAccurate.h`.
- SIMD layer (`-d:simd`, ADR-0005): dispatch at the umbrella and C ABI, never in
  `algorithms/` (vgraph back-edge). Compensated SIMD returns `(T, bool)`; the C
  ABI falls back to the scalar algorithm when `reliable = false`. FMA is used
  only in the dot product — the scalar `twoProductFMA` product split and the
  SIMD dot kernels' FMA intrinsics (ADR-0007 Lever 1); the compensated sum recurrence stays
  FMA-free (ADR-0004).
- `book/index.nim` is nimib: its code blocks are compiled and run at docs build,
  so prose that outlives its API breaks the build. `py/notebooks/quickstart.ipynb`
  plays the same role for Python and renders natively on GitHub.
- End covered sources with a blank line. Nim maps a trailing statement one line
  past EOF. `nimble coverage` suppresses exactly two lcov categories, both
  compiler artefacts with no source-level fix: `mismatch`, where lcov 2.x and
  gcov disagree on the end line of Nim's generated destructors, and that EOF + 1
  attribution -- `range` on lcov 2.5, `unmapped` on the 2.0 the runners install,
  which is why the task asks the version first. Every other error still fails.

## Scope

Numerical reduction engine: error-free transforms, accurate sums, dot products,
and policy-free statistical reductions. Apache-2.0, DCO.
