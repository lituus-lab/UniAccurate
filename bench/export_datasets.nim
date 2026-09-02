# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Export the canonical benchmark datasets ONCE, in a portable binary format read
## by every cross-backend comparator (the UniAccurate Nim driver, the C originals
## driver, the Rust `accurate` driver). Every backend reads the *same* on-disk
## bytes, so the inputs are bit-identical across columns and the comparison is
## real. Ported from the accuratesums reference harness, isolated under bench/.
##
## The datasets and per-combination seed are SHARED with the Nim driver via
## `bench_data` (same `genDataset`, same seed formula
## `0x5eed + n*131 + di*17 + typeTag.len`); the dot operand `y` uses a derived
## seed (`seed XOR 0x9e57`) so it is deterministic and paired with `x`.
##
## Binary format (little-endian): one file per `(type, dataset, n, operand)`
## holding `n` raw f64 (8 B) or f32 (4 B) values, contiguous, no padding. The
## manifest `bench/canonical/manifest.csv` lists every file with its element
## count and byte width; readers assert `filesize == n * bytes_per_elem`. The
## `.bin` and manifest are gitignored generated outputs (run `nimble benchData`),
## never committed.
##
## Pass `quick` for the 3-size subset used to verify the comparators without a
## multi-minute full run.
import std/[os, strutils]

import bench_data

const CanonicalDir = "bench/canonical"

proc writeRaw[T](path: string, x: openArray[T]) =
  ## Write `x` as raw native-endian bytes. amd64/arm64 are little-endian, so
  ## this is the portable little-endian encoding every reader expects.
  var f = open(path, fmWrite)
  defer: f.close()
  if x.len > 0:
    let nb = f.writeBuffer(unsafeAddr x[0], x.len * sizeof(T))
    doAssert nb == x.len * sizeof(T), "short write to " & path

proc seedFor(typeTag: string, di: int, n: int): int64 =
  ## Seed formula shared with the Nim driver: identical inputs.
  0x5eed'i64 + n.int64 * 131 + di.int64 * 17 + typeTag.len.int64

proc exportType(T: typedesc, typeTag: string, sizes: seq[int],
    manifest: var File) =
  for n in sizes:
    for di, ds in Datasets:
      let seedX = seedFor(typeTag, di, n)
      let seedY = seedX xor 0x9e57'i64 # paired dot operand, deterministic
      let x = genDataset(T, ds, n, seedX)
      let y = genDataset(T, ds, n, seedY)
      let stem = CanonicalDir / (typeTag & "_" & ds & "_" & $n)
      let xPath = stem & ".x.bin"
      let yPath = stem & ".y.bin"
      writeRaw(xPath, x)
      writeRaw(yPath, y)
      # type,dataset,n,bytes_per_elem,n_elems,x_file,y_file
      manifest.writeLine(typeTag & "," & ds & "," & $n & "," & $sizeof(T) &
          "," &
        $x.len & "," & xPath & "," & yPath)

proc run(quick: bool) =
  let sizes = if quick: @[10, 1_000, 100_000]
              else: @[10, 100, 1_000, 10_000, 100_000, 1_000_000]
  createDir(CanonicalDir)
  let manifestPath = CanonicalDir / "manifest.csv"
  var manifest = open(manifestPath, fmWrite)
  defer: manifest.close()
  manifest.writeLine("type,dataset,n,bytes_per_elem,n_elems,x_file,y_file")
  exportType(float64, "f64", sizes, manifest)
  exportType(float32, "f32", sizes, manifest)
  stderr.writeLine("Wrote " & manifestPath & " (" &
    $(2 * sizes.len * Datasets.len) & " dataset pairs).")

when isMainModule:
  let quick = paramCount() > 0 and paramStr(1) == "quick"
  run(quick)
