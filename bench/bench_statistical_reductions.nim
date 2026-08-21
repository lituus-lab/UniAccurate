# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import std/[algorithm, envvars, json, monotimes, os, times]
import UniAccurate

const
  PointCount = 100_000
  Warmups = 3
  Iterations = 20
  Runs = 3

func average(values: openArray[float64]): float64 =
  for value in values:
    result += value / values.len.float64

func median(values: openArray[float64]): float64 =
  var ordered = @values
  ordered.sort()
  ordered[ordered.len div 2]

template elapsed(body: untyped): float64 =
  block:
    let started = getMonoTime()
    body
    (getMonoTime() - started).inNanoseconds.float64 / 1_000_000.0

proc main() =
  let output = if paramCount() == 1: paramStr(1)
    else: "build/statistical-reductions-baseline.json"
  if paramCount() > 1:
    quit("usage: bench_statistical_reductions [output.json]", 2)
  var x = newSeq[float64](PointCount)
  var y = newSeq[float64](PointCount)
  for index in 0 ..< PointCount:
    x[index] = 1e10 + (index mod 1021).float64
    y[index] = -1e8 + (index mod 509).float64 * 0.5
  let centerX = 1e10 + 510.0
  let centerY = -1e8 + 127.0
  var meanRuns, normRuns, squaresRuns, crossRuns, cosineRuns: seq[float64]
  var guard: float64
  for _ in 0 ..< Runs:
    for _ in 0 ..< Warmups:
      guard += scaledMean(x)
      guard += scaledEuclideanNorm(x)
      guard += centeredSumSquares(x, centerX)
      guard += centeredCrossProduct(x, y, centerX, centerY)
      guard += centeredCosineSimilarity(x, y, centerX, centerY)
    var meanSamples, normSamples, squaresSamples, crossSamples,
      cosineSamples: seq[float64]
    for _ in 0 ..< Iterations:
      let meanElapsed = elapsed: guard += scaledMean(x)
      let normElapsed = elapsed: guard += scaledEuclideanNorm(x)
      let squaresElapsed = elapsed: guard += centeredSumSquares(x, centerX)
      let crossElapsed = elapsed:
        guard += centeredCrossProduct(x, y, centerX, centerY)
      let cosineElapsed = elapsed:
        guard += centeredCosineSimilarity(x, y, centerX, centerY)
      meanSamples.add meanElapsed
      normSamples.add normElapsed
      squaresSamples.add squaresElapsed
      crossSamples.add crossElapsed
      cosineSamples.add cosineElapsed
    meanRuns.add average(meanSamples)
    normRuns.add average(normSamples)
    squaresRuns.add average(squaresSamples)
    crossRuns.add average(crossSamples)
    cosineRuns.add average(cosineSamples)
  let report = %*{
    "provider": "UniAccurate",
    "version": UniAccurateVersion,
    "operation": "statistical-reductions",
    "compiler": NimVersion,
    "machine": getEnv("UNIACCURATE_BENCHMARK_MACHINE", hostCPU),
    "memory_manager": "orc",
    "point_count": PointCount,
    "warmups_per_run": Warmups,
    "iterations_per_run": Iterations,
    "runs": Runs,
    "mean_semantics": "exact total with quotient-scaled overflow fallback",
    "norm_semantics": "allocation-free scaled recurrence",
    "centered_semantics": "deviation allocation plus exact superaccumulator",
    "cosine_semantics": "normalized deviations plus exact dot and scaled norms",
    "mean_run_mean_ms": meanRuns,
    "norm_run_mean_ms": normRuns,
    "centered_squares_run_mean_ms": squaresRuns,
    "centered_cross_run_mean_ms": crossRuns,
    "centered_cosine_run_mean_ms": cosineRuns,
    "mean_median_ms": median(meanRuns),
    "norm_median_ms": median(normRuns),
    "centered_squares_median_ms": median(squaresRuns),
    "centered_cross_median_ms": median(crossRuns),
    "centered_cosine_median_ms": median(cosineRuns),
    "guard": guard
  }
  createDir(parentDir(output))
  writeFile(output, pretty(report) & "\n")
  echo pretty(report)

main()
