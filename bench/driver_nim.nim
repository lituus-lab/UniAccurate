# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## UniAccurate column of the cross-backend comparator: times the UniAccurate
## scalar sum + dot algorithms against the SAME canonical datasets the C
## originals and Rust `accurate` drivers read, and emits timing + candidate
## result bit-patterns. Ported from the accuratesums Nim driver, isolated under
## bench/ (never imported by src/).
##
## Datasets are regenerated in-process via `bench_data.genDataset` with the
## shared seed formula `0x5eed + n*131 + di*17 + typeTag.len` (and
## `seedY = seedX xor 0x9e57` for the dot operand) -- bit-identical to the
## `.bin` files `bench/export_datasets.nim` writes and the C/Rust drivers read,
## so every column sees the same inputs without cross-process file I/O.
##
## Adaptive timing (Google-Benchmark style): warm up, ramp m 10x until one block
## reaches ~5 ms, size the measured run to ~targetNs, report mean ns/call. A
## global `sinkAcc` (read at the end) forces the noSideEffect `func` results so
## they are not DCE'd.
##
## The `tag` CLI arg selects the output suffix (`nim`, `nim_fma`, `nim_simd`),
## so the FMA-on and SIMD builds (compiled with `-d:useFMA` / `-d:simd`) write
## distinct CSVs from this one source. SIMD-FMA dot kernels are timed under
## `defined(simd)` (added by the SIMD bench task).
##
## Output (bench/compare/ and bench/canonical/, gitignored -- generated data
## and results, not source):
##   perf_<tag>.csv   algo,dataset,n,type,time_ns,time_per_elem_ns,bw_gbs
##   cand_<tag>.csv   algo,dataset,n,type,bits   (uint64 IEEE bits; f32 widened
##                    exactly to f64, `type` selects ulp spacing)
import std/[os, monotimes, times, strutils]

import UniAccurate
import bench_data

# The `_fin` suffix marks the `assumeFinite = true` opt-in (lever 3: strips the
# isFin guard; for `neumaier` it also selects Neumaier's branched form, which
# folds to branchless without the guard and matches the C reference). `_fast`
# marks `useFastTwoSum = true` (lever 2: branched Dekker `twoSumFast` for the
# EFT swap). Both are bit-identical to the default on the bench datasets (none
# overflow under the opt-in), so their acc-matrix rows validate parity at n=1M.
# Nim-only: no C/Rust equivalent row (the C reference IS the branched form).
const
  SumAlgos = ["naive", "pairwise", "kahan", "neumaier", "klein", "shewchuk",
              "sum2", "sumK3", "neumaier_fin", "klein_fin", "sum2_fin",
              "sumK3_fast", "sumK3_fin"]
  DotAlgos = ["naive_dot", "dot2", "dotK3", "dot2_fin", "dotK3_fin"]

var sinkAcc: float64 = 0.0

proc timeNsPerCall(thunk: proc(), targetNs: float): float =
  ## Calibrate-then-measure: warm up, ramp m 10x until one timed block reaches
  ## ~5 ms, size the final run from that block, then time one clean block of
  ## ~targetNs and report its mean.
  thunk()
  var m = 1
  var dt = 0.0
  while true:
    let t0 = getMonoTime()
    for i in 0 ..< m: thunk()
    dt = float(inNanoseconds(getMonoTime() - t0))
    if dt >= 5e6 or m >= 1000: break
    m *= 10
  let per = if dt > 0: dt / m.float else: 1.0
  let iters = max(1, min(int(targetNs / per), 1 shl 22))
  let t0 = getMonoTime()
  for i in 0 ..< iters: thunk()
  result = float(inNanoseconds(getMonoTime() - t0)) / iters.float

proc callSum[T](name: string, x: openArray[T]): T {.inline.} =
  ## Dispatch to the named UniAccurate sum algorithm. `sumK3` is `sumK(K=3)`.
  ## The `_fin` variants pass `assumeFinite = true` (lever 3); `_fast` passes
  ## `useFastTwoSum = true` (lever 2). Both are bench-only measurement paths.
  case name
  of "naive": result = naiveSum(x)
  of "pairwise": result = pairwiseSum(x)
  of "kahan": result = kahanSum(x)
  of "neumaier": result = neumaierSum(x)
  of "klein": result = kleinSum(x)
  of "shewchuk": result = shewchukSum(x)
  of "sum2": result = sum2(x)
  of "sumK3": result = sumK(x, 3)
  of "neumaier_fin": result = neumaierSum(x, assumeFinite = true)
  of "klein_fin": result = kleinSum(x, assumeFinite = true)
  of "sum2_fin": result = sum2(x, assumeFinite = true)
  of "sumK3_fast": result = sumK(x, 3, useFastTwoSum = true)
  of "sumK3_fin": result = sumK(x, 3, assumeFinite = true)
  else: raise newException(ValueError, "unknown sum algorithm: " & name)

proc callDot[T](name: string, x, y: openArray[T]): T {.inline.} =
  ## Dispatch to the named UniAccurate dot algorithm. `dotK3` is `dotK(K=3)`.
  ## The `_fin` variants pass `assumeFinite = true` (lever 3), bench-only.
  case name
  of "naive_dot": result = naiveDot(x, y)
  of "dot2": result = dot2(x, y)
  of "dotK3": result = dotK(x, y, 3)
  of "dot2_fin": result = dot2(x, y, assumeFinite = true)
  of "dotK3_fin": result = dotK(x, y, 3, assumeFinite = true)
  else: raise newException(ValueError, "unknown dot algorithm: " & name)

proc bits[T](v: T): uint64 =
  ## IEEE bit pattern of `v` widened to f64 (f32 widens exactly; the `type`
  ## column tells the aggregator which ulp spacing to use).
  when T is float32:
    cast[uint64](float64(v))
  else:
    cast[uint64](v)

# EXPERIMENTAL (bench-only, `-d:naiveDotFMA`): a scalar dot that accumulates
# with one `fma(x, y, acc)` per element instead of `acc += x*y` (mul+add). One
# rounding per step, not two — NOT bit-identical to `naiveDot`, and not a library
# path: kept in bench/ to measure whether a single-rounded naive dot is worth a
# src/ algorithm (numbers + correctness decide at cherry-pick time).
when defined(naiveDotFMA):
  {.push checks: off.}
  proc libmFmaD(a, b, c: float64): float64 {.importc: "fma",
      header: "<math.h>".}
  proc libmFmafD(a, b, c: float32): float32 {.importc: "fmaf",
      header: "<math.h>".}
  {.pop.}

  func fmaDot[T: SomeFloat](x, y: openArray[T]): T =
    result = T(0)
    when T is float32:
      for i in 0 ..< x.len: result = libmFmafD(x[i], y[i], result)
    else:
      for i in 0 ..< x.len: result = libmFmaD(x[i], y[i], result)

proc runType(T: typedesc, typeTag: string, sizes: seq[int], targetNs: float,
             fPerf, fCand: File, totalCombos: int, done: var int) =
  for n in sizes:
    for di, ds in Datasets:
      let seedX = 0x5eed'i64 + n.int64 * 131 + di.int64 * 17 + typeTag.len.int64
      let seedY = seedX xor 0x9e57'i64
      let x = genDataset(T, ds, n, seedX)
      let y = genDataset(T, ds, n, seedY)
      let bytes = n.float * sizeof(T).float
      for algo in SumAlgos:
        let tns = timeNsPerCall(proc() = sinkAcc += float(callSum[T](algo, x)), targetNs)
        let perElem = tns / n.float
        let bw = if tns > 0: bytes / tns else: 0.0
        let val = callSum[T](algo, x)
        fPerf.writeLine(algo & "," & ds & "," & $n & "," & typeTag & "," &
          tns.formatFloat(ffDecimal, 3) & "," & perElem.formatFloat(ffDecimal,
              4) &
          "," & bw.formatFloat(ffDecimal, 2))
        fCand.writeLine(algo & "," & ds & "," & $n & "," & typeTag & "," &
            $bits(val))
        inc(done)
        stderr.writeLine("[" & $(done * 100 div totalCombos) & "%] " & algo &
            " " &
          typeTag & " " & ds & " n=" & $n)
      for algo in DotAlgos:
        let tns = timeNsPerCall(proc() = sinkAcc += float(callDot[T](algo, x,
            y)), targetNs)
        let perElem = tns / n.float
        let bw = if tns > 0: (2.0 * bytes) / tns else: 0.0 # x and y read
        let val = callDot[T](algo, x, y)
        fPerf.writeLine(algo & "," & ds & "," & $n & "," & typeTag & "," &
          tns.formatFloat(ffDecimal, 3) & "," & perElem.formatFloat(ffDecimal,
              4) &
          "," & bw.formatFloat(ffDecimal, 2))
        fCand.writeLine(algo & "," & ds & "," & $n & "," & typeTag & "," &
            $bits(val))
        inc(done)
        stderr.writeLine("[" & $(done * 100 div totalCombos) & "%] " & algo &
            " " &
          typeTag & " " & ds & " n=" & $n)
      when defined(simd):
        # SIMD dot kernels (FMA intrinsics): `naiveDotSimd` is an FMA reduce;
        # `dot2Simd`/`dotK3Simd` keep the ORO cascade vectorized. Dispatch is at
        # the umbrella — AVX-512/AVX2 (amd64) for f64, NEON (arm64) for f32,
        # scalar fallback off those ISAs — so the macOS arm64 column records NEON
        # f32 + scalar f64, and a Zen4 `-d:avx512` run records AVX-512 f64. The
        # `(T, bool)` kernels record `result[0]` (reliable is the C-ABI concern,
        # not throughput).
        block:
          let tn0 = timeNsPerCall(
            proc() = sinkAcc += float(naiveDotSimd[T](x, y)), targetNs)
          let pe0 = tn0 / n.float
          let bw0 = if tn0 > 0: (2.0 * bytes) / tn0 else: 0.0
          let v0 = naiveDotSimd[T](x, y)
          fPerf.writeLine("naive_dot_simd," & ds & "," & $n & "," & typeTag &
            "," &
            tn0.formatFloat(ffDecimal, 3) & "," & pe0.formatFloat(ffDecimal,
                4) &
            "," & bw0.formatFloat(ffDecimal, 2))
          fCand.writeLine("naive_dot_simd," & ds & "," & $n & "," & typeTag &
            "," &
            $bits(v0))
          let tn2 = timeNsPerCall(
            proc() = sinkAcc += float(dot2Simd[T](x, y)[0]), targetNs)
          let pe2 = tn2 / n.float
          let bw2 = if tn2 > 0: (2.0 * bytes) / tn2 else: 0.0
          let v2 = dot2Simd[T](x, y)[0]
          fPerf.writeLine("dot2_simd," & ds & "," & $n & "," & typeTag & "," &
            tn2.formatFloat(ffDecimal, 3) & "," & pe2.formatFloat(ffDecimal,
                4) &
            "," & bw2.formatFloat(ffDecimal, 2))
          fCand.writeLine("dot2_simd," & ds & "," & $n & "," & typeTag & "," &
            $bits(v2))
          let tn3 = timeNsPerCall(
            proc() = sinkAcc += float(dotK3Simd[T](x, y)[0]), targetNs)
          let pe3 = tn3 / n.float
          let bw3 = if tn3 > 0: (2.0 * bytes) / tn3 else: 0.0
          let v3 = dotK3Simd[T](x, y)[0]
          fPerf.writeLine("dotK3_simd," & ds & "," & $n & "," & typeTag & "," &
            tn3.formatFloat(ffDecimal, 3) & "," & pe3.formatFloat(ffDecimal,
                4) &
            "," & bw3.formatFloat(ffDecimal, 2))
          fCand.writeLine("dotK3_simd," & ds & "," & $n & "," & typeTag & "," &
            $bits(v3))
          inc(done, 3)
      when defined(naiveDotFMA):
        # EXPERIMENTAL single-rounded scalar dot — see `fmaDot` above. Recorded
        # as `naive_dot_fma`; its candidate diverges from the `naiveDot` oracle
        # by design (one rounding per step).
        let tn = timeNsPerCall(
          proc() = sinkAcc += float(fmaDot[T](x, y)), targetNs)
        let pe = tn / n.float
        let bw = if tn > 0: (2.0 * bytes) / tn else: 0.0
        let v = fmaDot[T](x, y)
        fPerf.writeLine("naive_dot_fma," & ds & "," & $n & "," & typeTag & "," &
          tn.formatFloat(ffDecimal, 3) & "," & pe.formatFloat(ffDecimal, 4) &
          "," & bw.formatFloat(ffDecimal, 2))
        fCand.writeLine("naive_dot_fma," & ds & "," & $n & "," & typeTag & "," &
          $bits(v))
        inc(done)

proc runAll(quick: bool, tag: string) =
  let sizes = if quick: @[10, 1_000, 100_000]
             else: @[10, 100, 1_000, 10_000, 100_000, 1_000_000]
  let targetNs = if quick: 0.05e9 else: 0.25e9
  createDir("bench/compare")
  let perfPath = "bench/compare/perf_" & tag & ".csv"
  let candPath = "bench/compare/cand_" & tag & ".csv"
  var fPerf = open(perfPath, fmWrite)
  var fCand = open(candPath, fmWrite)
  defer: fPerf.close(); fCand.close()
  fPerf.writeLine("algo,dataset,n,type,time_ns,time_per_elem_ns,bw_gbs")
  fCand.writeLine("algo,dataset,n,type,bits")
  let totalCombos = 2 * sizes.len * Datasets.len * (SumAlgos.len + DotAlgos.len)
  var done = 0
  runType(float64, "f64", sizes, targetNs, fPerf, fCand, totalCombos, done)
  runType(float32, "f32", sizes, targetNs, fPerf, fCand, totalCombos, done)
  stderr.writeLine("Wrote " & perfPath & " + " & candPath & " (" & $done &
    " rows). sink=" & $sinkAcc)

when isMainModule:
  let quick = paramCount() > 0 and paramStr(1) == "quick"
  let tag = if paramCount() > 1: paramStr(2) else: "nim"
  runAll(quick, tag)
