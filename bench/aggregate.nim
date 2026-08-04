# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Aggregate the per-backend bench CSVs into one comparison matrix.
##
## Reads each backend's `perf_<tag>.csv` (timing) and `cand_<tag>.csv` (candidate
## result bit-patterns) from bench/compare/, computes the correctly-rounded
## oracle ONCE per (type, dataset, n) with UniAccurate's `superSum` / `superDot`
## on the same canonical datasets (regenerated via `bench_data.genDataset` with
## the shared seed), and joins every backend's candidates against that single
## oracle. One oracle, all backends compared against it -- no MPFR, no per-
## backend oracle duplication.
##
## Outputs (bench/compare/, gitignored):
##   perf_matrix.csv   algo,dataset,n,type,backend,time_ns,time_per_elem_ns,bw_gbs
##   acc_matrix.csv    algo,dataset,n,type,backend,abs_err,rel_err,ulp_err
##   summary.md        ns/elem by algo x backend at the largest common n, per type
##
## A backend whose CSVs are absent is silently skipped (run only the columns you
## need; the matrix reports what was run). Run `nimble benchAll` after the bench
## tasks to produce the matrix.
import std/[os, math, strutils, strformat, tables, algorithm, sequtils, osproc]

import UniAccurate
import bench_data

## `--readme` also splices a condensed f64 table for a handful of headline
## algorithms into README.md, wrapped in a `<!-- bench:machine=<slug> -->`
## block. Re-running with the same slug replaces only that block; a
## different slug (`UNIACCURATE_BENCH_MACHINE` env var, for a box whose
## auto-detected slug is not the one you want recorded, e.g. a FreeBSD/Zen4
## box) appends alongside it -- one README can carry results from several
## machines without one overwriting another's.
const HeadlineAlgos = ["naive", "pairwise", "kahan", "dot2", "shewchuk"]

proc machineSlug(): string =
  if existsEnv("UNIACCURATE_BENCH_MACHINE"):
    return getEnv("UNIACCURATE_BENCH_MACHINE")
  var cpu = hostCPU
  when defined(macosx):
    let (brand, code) = execCmdEx("sysctl -n machdep.cpu.brand_string")
    if code == 0: cpu = brand.strip()
  elif defined(linux):
    let (model, code) = execCmdEx("sh -c \"grep -m1 'model name' /proc/cpuinfo | cut -d: -f2\"")
    if code == 0 and model.strip().len > 0: cpu = model.strip()
  result = (hostOS & "-" & cpu).toLowerAscii().multiReplace(
    (" ", "-"), ("(", ""), (")", ""), ("_", "-"))
  while "--" in result: result = result.replace("--", "-")

const OutDir = "bench/compare"
  # (backend label, file tag). The C/Rust drivers write fixed filenames; the
  # nimble tasks rename them to perf_<tag>.csv / cand_<tag>.csv before aggregating.
const Backends = [("nim", "nim"), ("nim_fma", "nim_fma"), ("nim_simd", "nim_simd"),
                  ("nim_fmadot", "nim_fmadot"), ("c", "c"), ("c_fma", "c_fma"),
                  ("rust", "rust"), ("rust_fma", "rust_fma")]
 # `dotK` is the Rust crate's `Dot3` row (written as `dotK`); the `_simd` /
 # `_fma` suffixes are the UniAccurate SIMD kernels (under `-d:simd`) and the
 # experimental single-rounded scalar dot (under `-d:naiveDotFMA`); `_fin` is
 # the `assumeFinite = true` opt-in (lever 3) — all dots, so all join against
 # the `superDot` oracle. The `_fin` membership is load-bearing: without it
 # `isDot` returns false and the dot would cross-contaminate with `superSum`.
const DotAlgos = ["naive_dot", "dot2", "dotK3", "dotK", "naive_dot_simd",
                  "dot2_simd", "dotK3_simd", "naive_dot_fma", "dot2_fin",
                  "dotK3_fin"]

proc isDot(algo: string): bool = algo in DotAlgos

proc ulpF64(v: float64): float64 =
  ## One ulp at |v| (the spacing of float64 at v's binade). Smallest subnormal
  ## for 0 / subnormals.
  if v == 0: return 5e-324
  let b = cast[uint64](abs(v))
  let exp = int(b shr 52) and 0x7FF
  if exp == 0: return 5e-324
  result = pow(2.0, float(exp - 1075)) # 2^(exp - bias - 52), bias = 1023

proc ulpF32(v: float32): float64 =
  ## One ulp at |v| for float32, returned as float64 for the error division.
  if v == 0: return 1.4e-45
  let b = cast[uint32](abs(v))
  let exp = int(b shr 23) and 0xFF
  if exp == 0: return 1.4e-45
  result = pow(2.0, float(exp - 150)) # 2^(exp - 127 - 23)

type Cand = tuple[algo, dataset, typ: string, n: int, bits: uint64]

proc parseCand(path: string): seq[Cand] =
  if not fileExists(path): return
  for line in lines(path):
    if line.len == 0 or line.startsWith("algo,"): continue
    let p = line.split(',')
    if p.len < 5: continue
    result.add((p[0], p[1], p[3], parseInt(p[2]), uint64(parseUInt(p[4]))))

type Perf = tuple[algo, dataset, typ: string, n: int, tns, perElem, bw: float]

proc parsePerf(path: string): seq[Perf] =
  if not fileExists(path): return
  for line in lines(path):
    if line.len == 0 or line.startsWith("algo,"): continue
    let p = line.split(',')
    if p.len < 7: continue
    result.add((p[0], p[1], p[3], parseInt(p[2]), parseFloat(p[4]),
               parseFloat(p[5]), parseFloat(p[6])))

proc seedFor(typeTag: string, di: int, n: int): int64 =
  0x5eed'i64 + n.int64 * 131 + di.int64 * 17 + typeTag.len.int64

proc oracle(typ: string, dataset: string, n: int, isDot: bool): float64 =
  ## Correctly-rounded oracle for the (type, dataset, n) cell. Sums use
  ## `superSum`; dots use `superDot`. Returned as float64 (f32 oracles are
  ## rounded to f32 first, then widened -- the value the f32 candidates compare
  ## against).
  let di = Datasets.find(dataset)
  let seedX = seedFor(typ, di, n)
  let seedY = seedX xor 0x9e57'i64
  if typ == "f32":
    let x = genDataset(float32, dataset, n, seedX)
    if isDot:
      let y = genDataset(float32, dataset, n, seedY)
      result = float64(superDot(x, y))
    else:
      result = float64(superSum(x))
  else:
    let x = genDataset(float64, dataset, n, seedX)
    if isDot:
      let y = genDataset(float64, dataset, n, seedY)
      result = superDot(x, y)
    else:
      result = superSum(x)

proc fmtErr(x: float64): string =
  if x == 0: "0" else: x.formatFloat(ffScientific, 4)

proc spliceReadme(slug: string, body: string) =
  ## Insert or replace the `<!-- bench:machine=slug -->` block in README.md.
  ## First run for a slug anchors on the `<!-- bench:insert -->` marker inside
  ## the Benchmarks section; a later run for the same slug replaces only the
  ## content between its own start/end tags.
  const path = "README.md"
  if not fileExists(path):
    stderr.writeLine("[readme] no README.md, skip splice")
    return
  let content = readFile(path)
  let startTag = "<!-- bench:machine=" & slug & " -->"
  let endTag = "<!-- /bench:machine=" & slug & " -->"
  let full = startTag & "\n" & body & "\n" & endTag
  if startTag in content:
    let s = content.find(startTag)
    let e = content.find(endTag) + endTag.len
    writeFile(path, content[0 ..< s] & full & content[e .. ^1])
    stderr.writeLine("[readme] replaced block for " & slug)
  else:
    const marker = "<!-- bench:insert -->"
    if marker notin content:
      stderr.writeLine("[readme] no <!-- bench:insert --> marker in README.md, skip splice")
      return
    let idx = content.find(marker) + marker.len
    writeFile(path, content[0 ..< idx] & "\n\n" & full & content[idx .. ^1])
    stderr.writeLine("[readme] inserted block for " & slug)

proc main() =
  if not dirExists(OutDir):
    stderr.writeLine("no bench/compare/ -- run the bench tasks first")
    quit(1)
  var fPerf = open(OutDir / "perf_matrix.csv", fmWrite)
  var fAcc = open(OutDir / "acc_matrix.csv", fmWrite)
  fPerf.writeLine("algo,dataset,n,type,backend,time_ns,time_per_elem_ns,bw_gbs")
  fAcc.writeLine("algo,dataset,n,type,backend,abs_err,rel_err,ulp_err")

  # Cache oracles by (typ,dataset,n,isDot) so each cell is computed once. The
  # `isDot` dimension is load-bearing: a sum and a dot share `(typ, ds, n)` but
  # take different oracles (`superSum` vs `superDot`), so omitting it would
  # cross-contaminate them — whichever algo ran first would cache its oracle for
  # the other.
  var oracleCache: Table[(string, string, int, bool), float64]
  proc getOracle(typ, ds: string, n: int, isDot: bool): float64 =
    let key = (typ, ds, n, isDot)
    if key notin oracleCache:
      oracleCache[key] = oracle(typ, ds, n, isDot)
    result = oracleCache[key]

  for (label, tag) in Backends:
    let cands = parseCand(OutDir / ("cand_" & tag & ".csv"))
    let perfs = parsePerf(OutDir / ("perf_" & tag & ".csv"))
    for p in perfs:
      fPerf.writeLine(p.algo & "," & p.dataset & "," & $p.n & "," & p.typ &
        "," &
        label & "," & p.tns.formatFloat(ffDecimal, 3) & "," &
        p.perElem.formatFloat(ffDecimal, 4) & "," & p.bw.formatFloat(ffDecimal, 2))
    for c in cands:
      let orc = getOracle(c.typ, c.dataset, c.n, isDot(c.algo))
      let val = cast[float64](c.bits) # f32 candidates are f64-widened f32
      let absErr = abs(val - orc)
      var relErr = 0.0
      var ulpErr = 0.0
      if orc != 0 and classify(orc) != fcInf and classify(orc) != fcNan:
        relErr = absErr / abs(orc)
        let ulp = if c.typ == "f32": ulpF32(float32(orc)) else: ulpF64(orc)
        ulpErr = absErr / ulp
      fAcc.writeLine(c.algo & "," & c.dataset & "," & $c.n & "," & c.typ & "," &
        label & "," & fmtErr(absErr) & "," & fmtErr(relErr) & "," & fmtErr(ulpErr))
    if cands.len == 0 and perfs.len == 0:
      stderr.writeLine("[aggregate] skip " & label & " (no CSVs)")
    else:
      stderr.writeLine("[aggregate] " & label & ": " & $perfs.len & " perf, " &
        $cands.len & " cand")

  # Close before the summary re-reads perf_matrix.csv below; otherwise the
  # last-written backends sit in the unflushed buffer and vanish from summary.md.
  fPerf.close()
  fAcc.close()

  # summary.md: ns/elem by algo x backend at the largest common n, per type.
  type Row = tuple[typ, algo: string, n: int]
  var bestN: Table[(string, string), int]
  var cell: Table[(string, string, int, string), float] # (typ,algo,n,backend)
  for line in lines(OutDir / "perf_matrix.csv"):
    if line.len == 0 or line.startsWith("algo,"): continue
    let p = line.split(',')
    if p.len < 8: continue
    let typ = p[3]; let algo = p[0]; let n = parseInt(p[2]); let be = p[4]
    let perElem = parseFloat(p[6])
    let k1 = (typ, algo)
    if n > bestN.getOrDefault(k1): bestN[k1] = n
    cell[(typ, algo, n, be)] = perElem
  var s = open(OutDir / "summary.md", fmWrite)
  defer: s.close()
  s.writeLine("# UniAccurate bench summary\n")
  s.writeLine("ns/elem at the largest common n per (type, algorithm). Lower is faster.\n")
  for typ in ["f64", "f32"]:
    s.writeLine("## " & typ & "\n")
    var algos: seq[string]
    for k, v in bestN.pairs:
      if k[0] == typ: algos.add(k[1])
    algos = deduplicate(algos)
    algos.sort()
    let backends = Backends.mapIt(it[0])
    s.write("| algo | n |" & backends.mapIt(" " & it & " |").join() & "\n")
    s.write("|---|---|" & backends.mapIt("---|").join() & "\n")
    for algo in algos:
      let n = bestN.getOrDefault((typ, algo))
      var row = "| " & algo & " | " & $n & " |"
      for be in backends:
        let v = cell.getOrDefault((typ, algo, n, be))
        row &= " " & (if v > 0: v.formatFloat(ffDecimal, 4) else: "-") & " |"
      s.writeLine(row)
    s.writeLine("")
  stderr.writeLine("Wrote " & OutDir / "perf_matrix.csv" & ", " &
    OutDir / "acc_matrix.csv" & ", " & OutDir / "summary.md")

  if paramCount() > 0 and paramStr(1) == "--readme":
    let backends = Backends.mapIt(it[0])
    var body = "All times ns/elem (f64, largest common n) -- lower is faster." &
      " `nim/c` is UniAccurate's default column against the honest external" &
      " C reference (below 1.0 would mean UniAccurate is faster). Full matrix" &
      " (all algorithms, f32 included): `nimble benchAll` locally, see" &
      " `bench/compare/summary.md` (generated, not tracked).\n\n"
    body &= "| algo | n |" & backends.mapIt(" " & it & " (ns/elem) |").join() &
      " nim/c |\n"
    body &= "|---|---|" & backends.mapIt("---|").join() & "---|\n"
    for algo in HeadlineAlgos:
      let n = bestN.getOrDefault(("f64", algo))
      if n == 0: continue
      var row = "| " & algo & " | " & $n & " |"
      for be in backends:
        let v = cell.getOrDefault(("f64", algo, n, be))
        row &= " " & (if v > 0: v.formatFloat(ffDecimal, 4) else: "-") & " |"
      let nimV = cell.getOrDefault(("f64", algo, n, "nim"))
      let cV = cell.getOrDefault(("f64", algo, n, "c"))
      row &= " " & (if nimV > 0 and cV > 0: (nimV / cV).formatFloat(ffDecimal, 2) & "x" else: "-") & " |"
      body &= row & "\n"
    const pyFrag = "bench/.md_python.md"
    if fileExists(pyFrag):
      body &= "\n**Python binding vs stdlib** (`nimble benchPython`)\n\n" & readFile(pyFrag)
    else:
      stderr.writeLine("[readme] no " & pyFrag & " -- run `nimble benchPython` too for the Python comparison")
    spliceReadme(machineSlug(), body)

when isMainModule:
  main()
