<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0007: Performance levers — isFin bit-mask, FMA enablement (incl. SIMD dot), assumeFinite, useFastTwoSum

- Status: Accepted
- Date: 2026-07-21
- Scope: `src/UniAccurate/twosum.nim` (`isFin`),
  `src/UniAccurate/algorithms/{orosum,compensatedsum}.nim` (opt-ins),
  `src/UniAccurate/simd.nim` (SIMD dot kernels), `config.nims` (FMA flags),
  `tests/test_{orosum,compensatedsum}.nim` (parity)

## Decision

Four performance levers, measured against the C originals and the Rust
`accurate` crate on a cross-backend harness (Zen4/FreeBSD amd64 and macOS
arm64). Three ship; one is a recorded negative result. Three of the `src/` levers
(`isFin`, `assumeFinite`, `useFastTwoSum`) are value-identical to the default on
finite non-overflowing input — pure hot-path switches, not divergent algorithms
(bit-identical for `isFin`/`assumeFinite`; value-identical for `useFastTwoSum`,
Invariant 1). The FMA lever is excluded from that invariant and described
separately in Lever 1: its `-d:useFMA`→`-mfma` enablement is bit-identical (same
C99 `fma()` result, faster codegen), but the experimental bench-only
`naiveDotFMA` single-rounded dot uses fused multiply-add with different rounding
than `naiveDot`, so it is faithful, not bit-identical.

### Lever 0 — `isFin` bit-mask (the root cause)

`isFin` ran `classify` (a full `fpclassify`, an external non-inlinable call)
once per element inside every compensated sum's hot loop. The 2–4× slowdown vs
the C originals on `kahanSum`/`neumaierSum`/`kleinSum` traced to **this guard,
not to EFT codegen**: with the `assumeFinite` opt-in (no guard) Nim already
matched C, proving the EFT recurrence inlines correctly.

Replaced with a bit-mask test (exponent field all-ones ⇔ NaN/±Inf): one `and`
plus one compare, fully inlinable, bit-identical semantics. This fixes the
**default** path — every caller benefits, no opt-in needed:
`kahan` 2.30 vs C 2.28, `klein` 1.47 vs C 1.54 (Zen4 f64, n=1M).

### Lever 1 — FMA enablement (scalar and SIMD dot)

`-d:useFMA` adds `-mfma` on amd64 (`config.nims`); arm64 NEON-FMA is base ISA
(no flag). With it, the scalar `twoProductFMA` `importc fma`/`fmaf` lowers to
one instruction instead of a libm call, and the SIMD dot kernels' FMA
intrinsics (`mm512_fmadd_pd`, `mm256_fmadd_pd`, `vfmaq_f32`) fire. The amd64
SIMD target flags now carry `-mfma` alongside `-mavx2`/`-mavx512f` — without it
clang refuses to inline the FMA intrinsics and `-d:simd -d:avx2` does not
build (every AVX2 CPU, Haswell+, has FMA, so the flag is portable).

ADR-0004's `-ffp-contract=off` stays in force: the FMA is the explicitly
called intrinsic / libm `fma`, never an `x*y+z` the compiler might fuse, so the
compensated *sum* recurrence stays FMA-free (ADR-0004 unchanged).

**Measured (Zen4 f64, n=1M, ns/elem):** on amd64 the FMA cliff is decisive —
`dot2` 24.5 → 1.70 with `-d:useFMA` (the libm `fma()` call dominates without
`-mfma`); the C originals show the same cliff (23.7 → 1.01). On arm64 there is
no cliff (NEON-FMA is base ISA): `dot2` 2.23 ≈ 2.24 either way. The AVX-512
SIMD dot kernels win outright: `dot2_simd` 0.31, `naive_dot_simd` 0.20 — 3.2×
over C `c_fma` (1.01), 7× over Rust. The experimental bench-only
`-d:naiveDotFMA` single-rounded scalar dot uses fused multiply-add
(`fma(x, y, acc)` — one rounding per element) where `naiveDot` uses `mul`+`add`
(two roundings): different rounding, so **faithful** (≤6 ulp at n=100, never
diverges), not bit-identical to `naiveDot`; measured only, never a library path.

#### SIMD dot kernel design (the ADR-0005 deferred piece)

ADR-0005 left the dot kernels for later and kept the SIMD layer FMA-free. The
dot kernels (`naiveDotSimd` / `dot2Simd` / `dotK3Simd`) vectorize the ORO
Dot2 / DotK recurrence per lane with FMA: each lane keeps a running product
sum `s` and an error cascade (`e` for Dot2, `(es, ec)` for DotK3), advanced by
`h = xᵢ·yᵢ`, `r₁ = fma(xᵢ, yᵢ, −h)` (the exact product error) and a vector
`twoSum` for the sum error `r₂`. The `L` lane K-fold estimates are merged
scalarly with `neumaierSum` (its `γ_K·Σ|xᵢyᵢ|` merge error is far below the
K-fold bound, so a 2-fold merge suffices). The tail is a zero-padded final
vector step so it stays K-fold without a scalar FMA. `naiveDotSimd` is an FMA
dot reduce with a naive scalar merge (within the naive bound, no reliability
flag).

This is **intentional FMA for the product**, not compiler contraction: the
FMA is the explicitly called intrinsic / `fma` libm call, never an `x*y+z`
the compiler might fuse. ADR-0004's prohibition on FMA inside the compensated
*sum* recurrence is unchanged — only the dot product's `xᵢ·yᵢ` and its error
extraction use FMA. Reliability follows the same `(T, bool)` convention as
the rest of the SIMD dot family (ADR-0006's fallback convention).

### Lever 3 — `assumeFinite` on the sum family (ships)

`kahanSum`/`neumaierSum`/`kleinSum` (`compensatedsum.nim`) and `sum2`/`sumK`
(`orosum.nim`) take `assumeFinite: static bool = false`, mirroring the
dot-family opt-in (ADR-0006). The opt-in strips the per-element `isFin` guard
(and, for the two-level schemes, the `isFin(t)` second-level guard and the
final `isFin(result)` compensation guard): the caller contracts that every
element is finite **and** no partial sum overflows, so the EFT recurrence runs
bare. The default (`false`) keeps the guards and is safe on arbitrary input.

Under `assumeFinite = true`, `neumaierSum`/`sum2` select **Neumaier's
magnitude-branched form** (`t = result + v`; `c += (result - t) + v` or
`(v - t) + result`) instead of the branchless `twoSum`. With the guard gone
the compiler folds the magnitude select to branchless code, matching the C
reference's exact shape. This **extends** ADR-0006's framing, which listed
`assumeFinite` only for the dot/oro slice; the compensated-sum recurrence is
the same EFT shape, so the opt-in applies identically.

**Measured (Zen4 f64, n=1M, ns/elem, `nim_fma`):**

| algo | default | `assumeFinite` | C `c_fma` | verdict |
|---|---|---|---|---|
| `neumaier` | 1.17 | 0.93 | 0.94 | C parity (~20% gain) |
| `sum2` | 1.17 | 0.92 | 1.74 | beats C (~20% gain) |
| `dot2` | 1.70 | 1.16 | 1.01 | ~32% gain; SIMD 0.31 wins |
| `dotK3` | 2.72 | 2.18 | — | ~20% gain |
| `klein` | 1.46 | 1.47 | 1.54 | neutral (EFT-bound, not guard) |
| `sumK3` | 3.74 | 3.88 | — | neutral (no branched form) |

The default `neumaier` stays at 1.17 (vs C 0.93) because the branchless
`twoSum` (6 ops, cancellation-robust) costs ~25% over C's branched form
(3 ops + branch); only the opt-in reclaims it. Trade-off documented: default =
robust to cancellation (a magnitude branch mispredicts as the running sum
flips sign, and the guard's control flow stops the compiler folding it
branchless); opt-in = C parity.

### Lever 2 — `useFastTwoSum` on `sumK` (negative result, kept)

`sumK` takes `useFastTwoSum: static bool = false`: when set, each EFT step
swaps Møller–Knuth `twoSum` (6 branchless FLOPs) for the branched Dekker
`twoSumFast` (3 FLOPs + one magnitude branch, requires `|a| >= |b|`, restored
by a branched swap). Both yield the same `(s, e)` EFT in *value* (`s + e == a + b`
exactly, `s = fl(a+b)`); only the sign of a zero error term may differ between
the two formulations, so the paths are **value-identical, not bit-identical**.
The parity test compares with `==` (under which `+0 == -0`), so it passes; a
raw-bit comparison would not.

**Measured: NOT a perf win.** The magnitude branch mispredicts on cancellation
data (the running sum flips sign) and does not fold better than the branchless
`twoSum`: `sumK3_fast` is marginal to negative vs `sumK3` (Zen4 `nim_fma`:
f64 ~−6%, f32 ~+2% — the f32 loss plus the mispredict make it not a win). The
parameter is **kept** for value-identical completeness and to consume the
previously-dead `twoSumFast`, but it is not recommended as a perf lever. Scope
honesty: only `transform`/`sumK` have a `twoSum` to swap; `sum2` delegates to
`neumaierSum`; `accSum`/`nearSum` use Rump's `rumpTransform`, which is built on
`fastTwoSum` (the 3-op preconditioned form) already — no lever there.

## Invariants

1. Every `src/` lever **except FMA** is **value-identical to the default on
   finite non-overflowing input** — bit-identical for `isFin` and `assumeFinite`
   (which run the same EFT), and value-identical for `useFastTwoSum` (only the
   sign of a zero error term may differ; see Lever 2). The FMA lever is
   excluded: its `-mfma` codegen enablement is bit-identical (same C99 `fma()`
   semantics), but the bench-only `naiveDotFMA` single-rounded dot uses fused
   multiply-add with different rounding than `naiveDot`'s `mul`+`add` — faithful
   (≤6 ulp), not bit-identical (Lever 1).
   Verified by parity tests (`tests/test_orosum.nim`,
   `tests/test_compensatedsum.nim`) that assert `f(..., opt = true) == f(..., opt
   = false)` (`==` treats `+0` and `-0` as equal) on the bounded/integer
   datasets, and re-confirmed at bench scale (n=1M, all six datasets: the
   `_fin`/`_fast` rows are 0 ulp vs the default on f64; f32 matches the
   default's own value).
   The opt-in-on-overflow output is undefined and is **not pinned** — only the
   guarded path's ±Inf result is asserted. The undefined behavior differs by
   whether the `assumeFinite` path still calls `twoSum` (whose finite-operand
   precondition raises `PreConditionDefect` in debug): `kleinSum`, `sumK`, and
   `transform` call `twoSum`, so on overflow they raise `PreConditionDefect` in
   debug and yield NaN/garbage in release; `kahanSum` (Kahan's `y = v - c` form),
   `neumaierSum` (the branched `t = result + v` form), and `sum2` (which
   delegates to `neumaierSum`) skip `twoSum` in their `assumeFinite` path, so
   they produce NaN in **both** debug and release — no precondition fires.
2. The `isFin` bit-mask is bit-identical to `classify` for every float class
   (zero, subnormal, normal, NaN, ±Inf): the exponent-all-ones encoding is
   exactly the NaN/Inf marker.
3. The `assumeFinite` contract is `require: not assumeFinite or allFin(x)` /
   `ensure: assumeFinite or (not allFin(x) or classify(result) != fcNan)` —
   the ensure's `assumeFinite or` short-circuits the NaN postcondition for the
   opt-in path (undefined on overflow), as the dot family.
4. FMA stays confined to the dot product (`xᵢ·yᵢ` and its error extraction) and
   the experimental bench-only `naiveDotFMA` — never the compensated sum
   recurrence (ADR-0004). `useFastTwoSum`/`assumeFinite` are FMA-free (they
   touch the addition EFT, which has no FMA form).
5. All opt-ins default off and compile away under `-d:release`; backward
   compatibility holds — every existing caller uses the defaulted single-arg
   form, the C ABI and SIMD dispatch are unaffected, `nimble testAll` /
   `ctest` stay green.

## References

- ADR-0004 — FMA contraction invariant (compensated sum recurrence FMA-free;
  `-ffp-contract=off`).
- ADR-0005 — SIMD layer (the dot kernels' FMA intrinsics land here, Lever 1).
- ADR-0006 — new algorithm suites (`dot2`/`dotK`/`sum2`/`sumK` originate
  there) and the fallback convention the SIMD dot kernels follow.
- Dekker, N.J. (1971). "A Floating-Point Technique for Extending the Available
  Precision." *Numer. Math.* 18, 224–242 — `twoSumFast` (the `|a| >= |b|`
  3-op EFT).
- Møller, O. (1965). "Quasi Double-Precision in Floating-Point Addition."
  *BIT* 5, 37–50 — `twoSum` (the 6-op branchless EFT).
- Graillat, D., Langlois, P., Louvet, N. (2006). "Choosing a Twice More
  Accurate Dot Product Implementation." ICNAAM 2006. HAL hal-01351480 —
  Dot2FMA, the FMA error extraction the SIMD dot kernels vectorize.
