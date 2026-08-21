<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0009: Statistical reduction primitives

- Status: Accepted
- Date: 2026-08-21

## Decision

UniAccurate provides centered sums of squares and cross-products as reduction
primitives, while UniStatistics owns estimator conventions and degrees of
freedom. Centered products use the exact superaccumulator after float
subtraction. Euclidean norm uses scaled sum-of-squares so a representable norm
does not overflow merely because an intermediate square would overflow.

## Consequences

Consumers share one stable numerical substrate without moving statistical
policy into UniAccurate. Non-finite values retain IEEE propagation. The centered
primitives allocate deviation buffers; the norm remains allocation-free.
