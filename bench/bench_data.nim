# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Shared dataset generators and algorithm dispatch for the UniAccurate
## benchmarks. Ported from the accuratesums reference harness (kept isolated in
## bench/, never imported by src/). Imported by `export_datasets.nim` (canonical
## inputs) and `driver_nim.nim` (the UniAccurate compare column), so every
## backend sees bit-identical inputs.
import std/[math, random]

import UniAccurate

const
  Algorithms* = ["naive", "pairwise", "kahan", "neumaier", "klein", "shewchuk"]
  Datasets* = ["uniform", "gaussian", "cancellation", "sorted", "shuffled", "extreme"]

proc callAlgo*(name: string, T: typedesc, x: openArray[T]): T {.inline.} =
  ## Dispatch to the named UniAccurate algorithm. Kept as a string-keyed `case`
  ## (not a proc table) so the compiler inlines the chosen body into each harness.
  case name
  of "naive":     result = naiveSum(x)
  of "pairwise":  result = pairwiseSum(x)
  of "kahan":     result = kahanSum(x)
  of "neumaier":  result = neumaierSum(x)
  of "klein":     result = kleinSum(x)
  of "shewchuk": result = shewchukSum(x)
  else: raise newException(ValueError, "unknown algorithm: " & name)

proc genDataset*[T: SomeFloat](t: typedesc[T], name: string, n: int, seed: int64): seq[T] =
  ## Deterministic dataset of `n` values of type `T`, seeded by `seed`. The six
  ## shapes exercise the regimes that matter for summation accuracy and cost:
  ##
  ##  uniform      — iid in [-1, 1]; the benign baseline.
  ##  gaussian     — Box-Muller standard normals; heavier tail, mild cancellation.
  ##  cancellation — `big + small - big` triples; the exact residual is `small`,
  ##                 but naive summation loses `small` (it rounds away below
  ##                 ulp(big)) — the catastrophic-cancellation stress case.
  ##  sorted       — ascending small positives; the worst case for naive
  ##                 forward-accumulation error growth.
  ##  shuffled     — a shuffled uniform; the "typical unsorted" regime.
  ##  extreme      — mantissa in [-1, 1] scaled by 2^e with e spanning the format's
  ##                 usable exponent range; mixed magnitudes, guard-bit stress.
  var r = initRand(seed)
  result = newSeq[T](n)
  template u2: float = (rand(r, 0.0 .. 1.0) - 0.5) * 2.0  # uniform in [-1, 1]
  case name
  of "uniform":
    for i in 0 ..< n: result[i] = T(u2())
  of "gaussian":
    # Box-Muller; pairs of independent standard normals.
    var i = 0
    while i < n:
      let u1 = max(rand(r, 0.0 .. 1.0), 1e-300)
      let u2v = rand(r, 0.0 .. 1.0)
      let mag = sqrt(-2.0 * ln(u1))
      result[i] = T(mag * cos(2.0 * PI * u2v))
      if i + 1 < n: result[i + 1] = T(mag * sin(2.0 * PI * u2v))
      inc(i, 2)
  of "cancellation":
    # big + small - big triples: the exact residual is `small`, but naive
    # summation loses `small` (it rounds away below ulp(big)).
    var i = 0
    while i < n:
      let big = T(pow(2.0, float(rand(r, 0 .. 40))))
      let s = T(u2())
      result[i] = big + s
      inc(i)
      if i < n: result[i] = -big; inc(i)
  of "sorted":
    # Ascending small positives: the worst case for naive forward-accumulation
    # error growth, and the case pairwise/blocking improves most.
    for i in 0 ..< n: result[i] = T(float(i + 1) / float(n))
  of "shuffled":
    for i in 0 ..< n: result[i] = T(u2())
    shuffle(r, result)
  of "extreme":
    # Mixed magnitudes: mantissa in [-1,1] scaled by 2^e (e spans the format's
    # usable exponent range), exercising cancellation and guard-bit paths.
    let eh = if T is float32: 12 else: 18
    for i in 0 ..< n:
      let e = int(rand(r, 0 .. (2 * eh))) - eh
      result[i] = T(u2() * pow(2.0, e.float))
  else: raise newException(ValueError, "unknown dataset: " & name)