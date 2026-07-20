<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0002: Apache License 2.0

- Status: Accepted
- Date: 2026-07-15
- Scope: UniAccurate

## Decision

UniAccurate is Apache-2.0. The explicit patent grant matters for a numerics
library expected to implement algorithms from published research: Apache-2.0
states plainly that using the implementation does not require a separate
patent license from contributors.

`NOTICE` records any bundled MIT-licensed dependency (`NimContracts`,
`nimsimd`) — their upstream copyright and license text travel with any
redistribution, alongside `LICENSE` and DCO sign-off on every commit
(`CONTRIBUTING.md`).
