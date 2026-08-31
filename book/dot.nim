# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniAccurate

nbInit(theme = useNimibook)
useLituus()
nb.title = "Dot product"

nbText: """
## Dot product

The dot product `xᵀy = Σ xᵢ·yᵢ` is the natural analogue of summation: each
product is split by an error-free transform (`twoProductFMA` → `(h, r₁)` with
`h + r₁ = xᵢ·yᵢ` exactly), and the per-term rounding errors — the product
error `r₁` and the sum error `r₂` from `twoSum` — are themselves accumulated
with a compensated sum. The total error is a final rounding of the result plus
a K-fold compensated remainder in the magnitude `E` — `O(u·|xᵀy| + u^K · E)` —
as if the dot were computed in *K-fold* working precision (twice for `K = 2`),
provided `(n−1)u < 1`:

    naiveDot: γ_{2n}(u) · E            left-to-right, two roundings per term
    dot2:     2u·|xᵀy| + 2·γ²_{n+1}(2u)·E   twice precision (Graillat Dot2FMA)
    dotK:     2u·|xᵀy| + 2·γ^K_{n+1}(2u)·E   K-fold (ORO Alg 5.3)

with `E = Σ|xᵢ·yᵢ|`. The `2u·|xᵀy|` term is the unavoidable final rounding; the
cascade term is `(n·2u)^K · E`, far below it for `K ≥ 2`. `dot2(x, y) ==
dotK(x, y, 2)` bit-for-bit; `K = 1` is the naive dot, `K < 1` treated as 1.

A product of two finite values can overflow to ±Inf (unlike a sum, whose
addends are the finite inputs), so opposite-sign ±Inf products can combine to
`+Inf + −Inf = NaN` — a summation-order artifact. On that rare overflow
`naiveDot` recovers the true value via `superDot` (the exact superaccumulator
dot), and `dot2` / `dotK` inherit the fallback through their `naiveDot`
overflow path, so finite inputs never yield NaN.

`dot2` / `dotK` take an `assumeFinite: static bool = false` opt-in that drops
the per-element finiteness / overflow guards — bit-identical to the guarded
path on finite non-overflowing input, and the hot path for benchmarks.
"""

nbCode:
  let dx = [1.0, 1e20, 1.0, -1e20]
  let dy = [1.0, 1.0, 1.0, 1.0]
  echo "naiveDot = ", naiveDot(dx, dy)
  echo "dot2     = ", dot2(dx, dy)
  echo "dotK K=3 = ", dotK(dx, dy, 3)

nbText: """
Under `-d:simd` the dot kernels vectorize the ORO Dot2 / DotK recurrence per
lane with FMA on AVX2 / AVX-512 (float64) and NEON (float32): each lane keeps a
running product sum and an error cascade, advanced by `h = xᵢ·yᵢ`,
`r₁ = fma(xᵢ, yᵢ, −h)` and a vector `twoSum` for `r₂`. The C ABI dispatches
through the kernel and falls back to the scalar body on a lane-concentration
guard (ADR-0007); NEON float64 has no SIMD path.

The C ABI exposes `ua_dot_naive` / `ua_dot2` / `ua_dot_k` and Python exposes
`naive_dot` / `dot2` / `dot_k`.

### References

- Ogita, T., Rump, S.M., Oishi, S. (2005). "Accurate Sum and Dot Product".
  *SIAM J. Sci. Comput.* 26(6), 1950–1988. doi:10.1137/S0036142903448029 —
  Algorithm 5.3 (DotK), Theorem 5.4 (K-fold bound).
- Graillat, D., Langlois, P., Louvet, N. (2006). "Choosing a Twice More
  Accurate Dot Product Implementation." ICNAAM 2006. HAL hal-01351480 —
  Dot2FMA, FMA error extraction.
- Graillat, D., Jézéquel, M. (2020). "Tight Interval Inclusions with
  Compensated Algorithms." *IEEE Trans. Comput.* 69(12), 1774–1783.
  doi:10.1109/TC.2019.2924005 — the CompDot / Dot2 bound.
"""

nbText: """
"""

nbSave
