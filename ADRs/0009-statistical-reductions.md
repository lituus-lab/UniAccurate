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

The scaled mean divides the exact rounded total when that total is
representable. Only when finite inputs overflow the total does it accumulate
`x[i] / n` exactly after each IEEE division. This keeps the fallback within the
range of the observations. It deliberately does not promise correctly rounded
rational division of the exact sum. Centered cosine similarity normalizes
deviations before its exact dot product, so correlation consumers do not form
overflowing sums of squares.
Centered projection coefficients combine normalized dot products and scale
ratios through binary mantissas and exponents. A representable least-squares
ratio therefore does not become `Inf / Inf` merely because both raw centered
moments exceed the result type's range.
`centeredNormState` retains the normalization scale and exact normalized
sum-of-squares without materialising `scale²`. `centeredSquaredRatio` consumes
that state for repeated regression leverage evaluation. This Nim substrate
lets UniStatistics retain a safe fit without duplicating arithmetic; C and
Python consumers reach the operation through UniStatistics regression APIs.

## Consequences

Consumers share one stable numerical substrate without moving statistical
policy into UniAccurate. Non-finite values retain IEEE propagation. The centered
primitives allocate deviation buffers; the norm remains allocation-free.
The scaled mean is allocation-free. Centered cosine similarity allocates two
normalized deviation buffers and returns NaN when either centered vector is
zero, leaving constant-sample policy to its statistical consumer.
