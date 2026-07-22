<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0004: FMA contraction invariant

- Status: Accepted
- Date: 2026-07-20
- Scope: UniAccurate

## Decision

The C compiler must never rewrite `a*b - c` into a fused `fma(a, b, -c)`. That
contraction breaks the Dekker error identity of `twoProduct`: the
`ah*bh - result[0]` term must round separately from the product, or the
recovered error `e` is wrong. `twoProductFMA` uses the C99 libm `fma`/`fmaf` (a
single correctly-rounded op), never a compiler-contracted expression, so the
guard does not affect it.

`-ffp-contract=off` is applied for GCC and Clang. Clang contracts by default
(`-ffp-contract=on` fuses within a single statement; effective at all `-O` on
Clang ≥ 14, and at `-O1+` on older Clang), so the guard is required there. GCC
does not contract by default (`-ffp-contract=off`); the guard is defensive.
MSVC's `/fp:precise` already forbids contraction and does not accept the flag,
so MSVC builds (`--cc:vcc`) are excluded.

`-mfma` is a separate, orthogonal concern — performance, not correctness. It
inlines `fma`/`fmaf` to one hardware instruction but assumes the target CPU has
FMA; it is opt-in (`-d:useFMA`), default off, so a generic amd64 wheel cannot
SIGILL on a non-FMA CPU. libm `fma` stays correctly-rounded with or without it.

## Invariants

1. `-ffp-contract=off` on every GCC/Clang build path (amd64, arm64, others).
2. `-mfma` is enabled only by `-d:useFMA` (amd64) or by `-d:simd` with
   `-d:avx2`/`-d:avx512` (every AVX2 CPU, Haswell+, has FMA, so the SIMD target
   flags carry `-mfma`); never on a default scalar build.
3. `-d:scalarUniAccurate` opts out of both for a scalar, flag-free build.
4. MSVC builds receive neither flag; `/fp:precise` is the invariant there.
