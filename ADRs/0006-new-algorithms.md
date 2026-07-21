<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0006: New algorithm suites and fallback convention

- Status: Accepted
- Date: 2026-07-21
- Scope: UniAccurate `src/UniAccurate/algorithms/{shewchuksum,exactsum,orosum,dotproduct}.nim`,
  `src/UniAccurate/simd.nim` (dot kernels), the C ABI, the Python binding, and
  the `-d:simd` dispatch path

## Decision

Four algorithm families join the library, each a vertical slice (algorithm →
tests → C ABI → Python → docs), sourced as a clean rewrite of an external
reference (never named in code or commits — only this ADR and the plan record
the inspiration):

1. **`shewchukSum`** — the CPython `math.fsum` recipe (Hettinger over
   Shewchuk 1997): a non-overlapping expansion accumulated by `twoSum`,
   collapsed round-half-to-even. Correctly rounded (bit-identical to
   `math.fsum` on float64). No FMA, no SIMD kernel.
2. **`superSum` / `superDot`** — Radford Neal's small superaccumulator
   (arXiv:1505.05571): integer exact accumulation, lazy carry, a 64×64→128
   product step for `superDot`. Order-invariant, correctly rounded. No FMA, no
   SIMD kernel. `superDot` is the finite-input NaN fallback for the dot family.
3. **`sum2` / `sumK` / `accSum` / `nearSum` / `conditionNumber`** —
   Ogita–Rump–Oishi (2005) and Rump (2008): the online cascaded transform
   (`sum2` delegates to `neumaierSum`; `sumK` is the O(1)-space single-pass
   `VecSum` distillation), Rump's faithful `accSum` and correctly-rounded
   `nearSum`, and the `cond(sum)` diagnostic. No FMA, no SIMD kernel.
4. **`naiveDot` / `dot2` / `dotK`** — ORO Algorithm 5.3 via `twoProductFMA` +
   `twoSum`; `dotK` is the online (K−1)-fold cascade. FMA and SIMD kernels
   ship for this family.

### Fallback convention

Two coexisting reliability styles, both already present in the repo:

- **Scalar new algorithms** — bare `T` return, with an internal fallback to
  `naiveSum` / `naiveDot` (and through it `superSum` / `superDot`) on
  non-finite input or an intermediate overflow. Matches the existing scalar
  `naiveSum` / `kahanSum`. No `(T, bool)`: the fallback is inside the body, so
  the result is always safe to return directly.
- **SIMD dot kernels** — `(T, bool)` with
  `reliable = isFin(r) and maxLane <= LaneConcentrationFallback * abs(r)`, and
  the C ABI falls back to the scalar algorithm when `reliable = false` (the
  ADR-0005 idiom, extended to the dot kernels).

The dot family has one extra subtlety a sum does not: a product of two finite
values can overflow to ±Inf, so opposite-sign ±Inf products can combine to
`+Inf + −Inf = NaN` — a summation-order artifact (the exact dot of finite
values is finite or a correctly-signed ±Inf, never NaN). `naiveDot` recovers
via `superDot` when its result is NaN but every input is finite; `dot2` /
`dotK` inherit the recovery through their `naiveDot` overflow path. The SIMD
`naiveDotSimd` does *not* carry this internally, so the C ABI applies the same
`superDot` fallback after the SIMD path — the finite-input-never-NaN
postcondition holds at the C ABI for the whole family.

### SIMD coverage of the new families

Only the dot family gets SIMD kernels. `shewchukSum` (sequential expansion),
`superSum` / `superDot` (integer superaccumulator), and the ORO/Rump cascade
are sequential by nature; under `-d:simd` the C ABI dispatches them to the
scalar algorithm (documented in the C header comments and the book). NEON
float64 has no SIMD path, so `ua_dot_*` stay scalar on arm64 even with
`-d:simd`; the NEON float32 dot kernels are exercised through the Nim tests.

## Invariants

1. The new `algorithms/*.nim` modules auto-assigned the `algorithms` vgraph
   layer may import `twosum` and each other, never `simd` or `c_api`.
   `nimble checkVGraph` stays green: `simd` → `algorithms` is downward, the
   dot kernels dispatch at the C ABI / umbrella, not inside `dotproduct.nim`.
2. Scalar new algorithms return bare `T` with an internal fallback; SIMD dot
   kernels return `(T, bool)` and the C ABI falls back to scalar on
   `reliable = false`.
3. `superDot` is the finite-input NaN fallback for the whole dot family
   (scalar `naiveDot` directly; `dot2` / `dotK` and the SIMD path through it).
4. The C ABI never raises; it clamps out-of-range `csize_t` to `high(int)`
   before narrowing (the pattern from the existing `ua_sum_*`).
5. `shewchukSum` / `superSum` / `superDot` / the ORO–Rump family have no SIMD
   kernel — scalar under `-d:simd` (documented). Only `naiveDot` / `dot2` /
   `dotK` dispatch through SIMD kernels, and only on AVX2/AVX-512 (float64) or
   NEON (float32).

## References

- Shewchuk, J.R. (1997). "Adaptive Precision Floating-Point Arithmetic and Fast
  Robust Geometric Predicates." *Discrete & Computational Geometry* 18,
  305–363 — the expansion arithmetic; Hettinger's CPython `math.fsum` recipe
  is the direct source for `shewchukSum`.
- Neal, R. (2015). "Fast Exact Summation Using Small and Large
  Superaccumulators." arXiv:1505.05571 — the small superaccumulator
  (`superSum` / `superDot`).
- Ogita, T., Rump, S.M., Oishi, S. (2005). "Accurate Sum and Dot Product."
  *SIAM J. Sci. Comput.* 26(6), 1950–1988. doi:10.1137/S0036142903448029 —
  `sum2` (Alg 4.1), `sumK` (Alg 4.8), `dot2` / `dotK` (Alg 5.3, Thm 5.4).
- Rump, S.M. (2008). "Accurate summation." *Numer. Math.* 110, 385–404.
  doi:10.1007/s00211-008-0144-z — `accSum` (Alg 4.5), `nearSum` (Alg 7.4).
