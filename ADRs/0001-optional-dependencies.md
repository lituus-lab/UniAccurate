<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0001: Optional infra dependencies stay optional

- Status: Accepted
- Date: 2026-07-15
- Scope: UniAccurate

## Decision

UniAccurate depends on no other library of its own kind. Its two
dependencies are build/verification infra, not domain code, and are treated
differently:

- **nimsimd** (SIMD intrinsics bindings) is strictly optional: a scalar build
  (no `-d:simd`) never imports it, and the SIMD layer is an empty module
  without the flag. It never becomes a hard edge — `nimble checkVGraph` fails
  the build if a scalar-path module ever imports it.
- **NimContracts** (design-by-contract macros: `require:`/`ensure:`/`body:`)
  is a real build-time dependency — it is not gated behind a flag — but its
  runtime checks compile away entirely under `-d:release`, so it costs
  nothing in the artifact that ships.

## Invariants

1. Every module under the algorithm layer stays importable with `nimsimd`
   absent from the search path when `-d:simd` is not passed.
2. A `-d:release` build has zero `NimContracts` runtime overhead: contracts
   compile to nothing, verified by `nimble testRelease`.
3. `vgraph.cfg`'s `[engines]` section stays empty; `nimble checkVGraph`
   fails the build the day this repo gains a dependency on another
   similarly-prefixed package without that section being updated first.
