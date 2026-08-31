# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# UniAccurate — error-free transforms and accurate summation/dot-product.

version = "1.1.0"
author = "lituus-lab"
description = "Error-free transforms and accurate summation/dot-product (Nim + C-ABI + Python)"
license = "Apache-2.0"
srcDir = "src"

requires "nim >= 2.0.0"
requires "https://github.com/lbartoletti/NimContracts#main"
# SIMD backend (AVX-512 + NEON), gated by `-d:simd`.
requires "https://github.com/lbartoletti/nimsimd#master"

# nimble 0.22 exits 0 even when an `exec` inside a task fails, so a task's exit
# code says nothing about whether its body ran. Each task writes a marker as
# its last statement; `tools/gate.nim` removes the marker, runs the task, and
# fails if it is not there afterwards. `nimble canary` proves the gate still
# bites -- if that one ever passes, every other green result is worthless.
const gateExe =
  when defined(windows): "build/unigate.exe" else: "build/unigate"

template done(task: string) =
  mkDir "build/.gate"
  writeFile("build/.gate/" & task & ".ok", "")

proc gate(task: string): string =
  ## `exec gate("test")` -- builds the tool on first use.
  if not fileExists(gateExe):
    exec "nim c --hints:off -o:" & gateExe & " tools/gate.nim"
  gateExe & " " & task

task canary, "Must fail: proves the gate still catches a broken build":
  # No `done` here on purpose: the exec below raises, so the marker is never
  # written and the gate reports the failure nimble swallowed.
  exec "nim c -r --hints:off --path:src -o:build/canary tests/canary_broken.nim"

task lint, "Fail if nimpretty would reformat a source":
  exec "nim c -r --hints:off -o:build/lint_tool tools/lint.nim"
  done "lint"

task checkVGraph, "Fail on an import that climbs the layers in vgraph.cfg":
  exec "nim c -r --hints:off -o:build/vgraph_tool tools/vgraph.nim"
  done "checkVGraph"

# From the URL with a tag, not from the registry: the nimble registry lags
# upstream, and `nimble install nimibook` resolves 0.3.1, whose themes.nim does
# not compile against nimib 0.4.x.
const bookDeps = [
  "https://github.com/pietroppeter/nimib#v0.4.1",
  "https://github.com/pietroppeter/nimibook#v0.4.0",
  "https://github.com/lituus-lab/lituus-theme#v0.2.0",
]
taskRequires "docsDeps", bookDeps[0], bookDeps[1], bookDeps[2]
taskRequires "book", bookDeps[0], bookDeps[1], bookDeps[2]
taskRequires "docs", bookDeps[0], bookDeps[1], bookDeps[2]

task docsDeps, "Install the docs toolchain (nimib + nimibook + theme)":
  echo "nimib, nimibook and lituus-theme installed."
  done "docsDeps"

task bookInit, "Scaffold a chapter added to the table of contents":
  withDir "book":
    exec "nim c -r --hints:off -o:../build/nbook nbook.nim init"
  done "bookInit"

task book, "Build the multi-chapter book (needs nimib + nimibook)":
  withDir "book":
    exec "nim c -r --hints:off -o:../build/nbook nbook.nim clean"
    # `init` before `build`, on every run: it is what creates `__site/assets`,
    # which is not tracked, so a fresh clone has none and every page ships
    # referencing a stylesheet and a script that are not there.
    exec "nim c -r --hints:off -o:../build/nbook nbook.nim init"
    exec "nim c -r --hints:off -o:../build/nbook nbook.nim build"
  done "book"

task docs, "API reference + book into pages/ — what CI publishes":
  rmDir "pages"
  exec gate("book")
  cpDir "book/__site", "pages"
  # book.json is nimibook's build state -- no page fetches it -- and it carries
  # the absolute path of the machine that built it.
  rmFile "pages/book.json"
  exec "nim doc --index:on --outdir:pages/api --project --hints:off src/UniAccurate.nim"
  # ...and the reference wears the same theme. `nim doc` has no stylesheet
  # option, so the palette is appended to the one it just wrote.
  exec "nim c -r --hints:off --outdir:build tools/theme_api.nim " &
       "pages/api/nimdoc.out.css"
  done "docs"

task eft, "EFT tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_eft tests/test_eft.nim"
  exec "nim c -r --path:src -o:build/test_version tests/test_version.nim"
  done "eft"

task sum, "Summation tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_naive tests/test_naivesum.nim"
  exec "nim c -r --path:src -o:build/test_pairwise tests/test_pairwisesum.nim"
  done "sum"

task comp, "Compensated summation tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_comp tests/test_compensatedsum.nim"
  done "comp"

task shewchuk, "Shewchuk summation tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_shewchuk tests/test_shewchuksum.nim"
  done "shewchuk"

task exact, "Exact superaccumulator tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_exact tests/test_exactsum.nim"
  done "exact"

task oro, "ORO/Rump summation tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_oro tests/test_orosum.nim"
  done "oro"

task dot, "Dot product tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_dot tests/test_dotproduct.nim"
  done "dot"

task statisticalReductions, "Statistical reduction tests":
  exec "nim c -r --path:src -o:build/test_statistical_reductions" &
       " tests/test_statistical_reductions.nim"
  done "statisticalReductions"

task dispatch, "Runtime AVX2/AVX-512 dot dispatch tests (ADR-0008)":
  # -d:release, not debug: simd_dispatch.nim's {.raises: [].} on its scalar
  # fallback wrappers only holds once NimContracts' require/ensure on
  # naiveDot/dot2/dotK compile away (AGENTS.md) -- the real c_api.nim build is
  # always -d:release, so this test exercises the module the way it ships.
  exec "nim c -r -d:release --path:src -o:build/test_dispatch tests/test_simd_dispatch.nim"
  done "dispatch"

task expansions, "Shewchuk expansion tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_expansions tests/test_expansions.nim"
  done "expansions"

task prop, "Randomized property tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_prop tests/test_property.nim"
  done "prop"

task test, "Nim tests (debug, contracts active)":
  exec gate("eft")
  exec gate("sum")
  exec gate("comp")
  exec gate("shewchuk")
  exec gate("exact")
  exec gate("oro")
  exec gate("dot")
  exec gate("statisticalReductions")
  exec gate("dispatch")
  exec gate("expansions")
  exec gate("prop")
  done "test"

task testRelease, "Nim tests (release, contracts compiled away)":
  exec "nim c -r -d:release --path:src -o:build/test_eft_rel tests/test_eft.nim"
  exec "nim c -r -d:release --path:src -o:build/test_naive_rel tests/test_naivesum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_pairwise_rel tests/test_pairwisesum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_comp_rel tests/test_compensatedsum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_shewchuk_rel tests/test_shewchuksum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_exact_rel tests/test_exactsum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_oro_rel tests/test_orosum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_dot_rel tests/test_dotproduct.nim"
  exec "nim c -r -d:release --path:src -o:build/test_statistical_reductions_rel" &
       " tests/test_statistical_reductions.nim"
  exec "nim c -r -d:release --path:src -o:build/test_dispatch_rel tests/test_simd_dispatch.nim"
  exec "nim c -r -d:release --path:src -o:build/test_expansions_rel tests/test_expansions.nim"
  exec "nim c -r -d:release --path:src -o:build/test_prop_rel tests/test_property.nim"
  done "testRelease"

task testCi, "Nim tests (CI subset, debug)":
  exec gate("eft")
  exec gate("sum")
  exec gate("comp")
  exec gate("shewchuk")
  exec gate("exact")
  exec gate("oro")
  exec gate("dot")
  exec gate("statisticalReductions")
  exec gate("dispatch")
  exec gate("expansions")
  exec gate("prop")
  done "testCi"

task testCiRelease, "Nim tests (CI subset, release)":
  exec "nim c -r -d:release --path:src -o:build/test_eft_rel tests/test_eft.nim"
  exec "nim c -r -d:release --path:src -o:build/test_naive_rel tests/test_naivesum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_pairwise_rel tests/test_pairwisesum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_comp_rel tests/test_compensatedsum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_shewchuk_rel tests/test_shewchuksum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_exact_rel tests/test_exactsum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_oro_rel tests/test_orosum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_dot_rel tests/test_dotproduct.nim"
  exec "nim c -r -d:release --path:src -o:build/test_statistical_reductions_rel" &
       " tests/test_statistical_reductions.nim"
  exec "nim c -r -d:release --path:src -o:build/test_dispatch_rel tests/test_simd_dispatch.nim"
  exec "nim c -r -d:release --path:src -o:build/test_expansions_rel tests/test_expansions.nim"
  exec "nim c -r -d:release --path:src -o:build/test_prop_rel tests/test_property.nim"
  done "testCiRelease"

task testAll, "debug + release + C ABI":
  exec gate("test")
  exec gate("testRelease")
  exec gate("ctest")
  done "testAll"

task example, "Nim demo":
  exec "nim c -r --path:src -o:build/demo examples/demo.nim"
  done "example"

task bench, "Quick scalar bench smoke (release, reduced sizes) -- not the AVX-512 reference numbers":
  exec "nim c -r -d:release --path:src -o:build/bench_driver bench/driver_nim.nim quick ci"
  done "bench"

task benchStatisticalReductions, "Three-run statistical reduction baseline":
  exec "nim c -r -d:release --mm:orc --path:src" &
       " -o:build/bench_statistical_reductions" &
       " bench/bench_statistical_reductions.nim"
  done "benchStatisticalReductions"

# Nim takes `-o:` literally and appends no platform extension.
const
  sharedLib =
    when defined(windows): "libUniAccurate.dll"
    elif defined(macosx): "libUniAccurate.dylib"
    else: "libUniAccurate.so"
  staticLib = "libUniAccurate.a" # MinGW `ar` on Windows, so `.a` everywhere.

  # @rpath install_name, so the copy bundled in the wheel is found at import.
  macArgs =
    when defined(macosx): " --passL:\"-Wl,-install_name,@rpath/" & sharedLib & "\""
    else: ""

  # Host amd64 defaults to AVX2; arm64 needs no flag. config.nims adds the -m.
  simdArch =
    when defined(amd64): " -d:avx2"
    else: ""

task clib, "C shared library":
  exec "nim c --app:lib --noMain --mm:arc -d:release -o:" & sharedLib & macArgs &
       " src/UniAccurate/c_api.nim"
  done "clib"

task clibStatic, "C static library":
  exec "nim c --app:staticlib -d:staticNoAutoInit --noMain --mm:arc -d:release -o:" & staticLib &
       " src/UniAccurate/c_api.nim"
  done "clibStatic"

task clibMsvc, "C static library, MSVC ABI (Windows Python extension)":
  # CPython on Windows is MSVC-built and cannot link MinGW output.
  exec "nim c --cc:vcc --app:staticlib -d:staticNoAutoInit --noMain --mm:arc -d:release" &
       " -o:UniAccurate.lib src/UniAccurate/c_api.nim"
  done "clibMsvc"

# Nim's MinGW toolchain names it mingw32-make.
let makeExe = if findExe("mingw32-make").len > 0: "mingw32-make" else: "make"

# Windows-only variable overrides, passed on the make command line (wins over
# the Makefiles' own `?=` defaults on every make flavor) instead of an
# in-Makefile OS conditional -- GNU make, BSD make (FreeBSD's default `make`),
# and mingw32-make share no common `ifeq`/`.if` directive syntax.
const makeWinArgs =
  when defined(windows):
    " CC=gcc BIN=test_uniaccurate.exe RUN=test_uniaccurate.exe" &
    " RM_F=\"del /q\" LIBS="
  else: ""
const makeWinArgsExample =
  when defined(windows):
    " CC=gcc BIN=demo.exe RUN=demo.exe RM_F=\"del /q\" LIBS="
  else: ""

# `make -C`, not `cd dir && make`: nimble's exec runs no shell on Windows.
task ctest, "C ABI tests":
  exec gate("clibStatic")
  exec makeExe & " -C tests/c" & makeWinArgs
  done "ctest"

task cexample, "C demo":
  exec gate("clibStatic")
  exec makeExe & " -C examples/c" & makeWinArgsExample
  done "cexample"

task simd, "Run test_simd with -d:simd (host-default ISA)":
  exec "nim c -r -d:simd --path:src -o:build/test_simd tests/test_simd.nim"
  done "simd"

task simdAvx2, "Run test_simd with AVX2 (amd64)":
  exec "nim c -r -d:simd -d:avx2 --path:src -o:build/test_simd_avx2 tests/test_simd.nim"
  done "simdAvx2"

task simdAvx512, "Run test_simd with AVX-512 (amd64 Zen4)":
  exec "nim c -r -d:simd -d:avx512 --path:src -o:build/test_simd_avx512 tests/test_simd.nim"
  done "simdAvx512"

task testSimd, "Scalar Nim tests compiled under -d:simd (host ISA)":
  exec "nim c -r -d:simd" & simdArch &
       " --path:src -o:build/test_eft_simd tests/test_eft.nim"
  exec "nim c -r -d:simd" & simdArch &
       " --path:src -o:build/test_naive_simd tests/test_naivesum.nim"
  exec "nim c -r -d:simd" & simdArch &
       " --path:src -o:build/test_pairwise_simd tests/test_pairwisesum.nim"
  exec "nim c -r -d:simd" & simdArch &
       " --path:src -o:build/test_comp_simd tests/test_compensatedsum.nim"
  exec "nim c -r -d:simd" & simdArch &
       " --path:src -o:build/test_shewchuk_simd tests/test_shewchuksum.nim"
  exec "nim c -r -d:simd" & simdArch &
       " --path:src -o:build/test_exact_simd tests/test_exactsum.nim"
  exec "nim c -r -d:simd" & simdArch &
       " --path:src -o:build/test_oro_simd tests/test_orosum.nim"
  exec "nim c -r -d:simd" & simdArch &
       " --path:src -o:build/test_dot_simd tests/test_dotproduct.nim"
  exec "nim c -r -d:simd" & simdArch &
       " --path:src -o:build/test_expansions_simd tests/test_expansions.nim"
  exec "nim c -r -d:simd" & simdArch &
       " --path:src -o:build/test_prop_simd tests/test_property.nim"
  done "testSimd"

task clibSimd, "C shared library with -d:simd (host ISA)":
  exec "nim c --app:lib --noMain --mm:arc -d:release -d:simd" & simdArch &
       " -o:" & sharedLib & macArgs & " src/UniAccurate/c_api.nim"
  done "clibSimd"

task ctestSimd, "C ABI tests with -d:simd (host ISA)":
  exec "nim c --app:staticlib -d:staticNoAutoInit --noMain --mm:arc -d:release -d:simd" & simdArch &
       " -o:" & staticLib & " src/UniAccurate/c_api.nim"
  exec makeExe & " -C tests/c" & makeWinArgs
  done "ctestSimd"

task pyDeps, "Install Python build deps (setuptools, Cython, pytest) if missing":
  exec "python3 -m pip install --break-system-packages --quiet setuptools wheel \"Cython>=3.0.0\" pytest"
  done "pyDeps"

# The extension links the vcc static lib on Windows, the shared lib elsewhere.
task pyLib, "Build the library the Python extension links against":
  when defined(windows):
    exec gate("clibMsvc")
  else:
    exec gate("clib")
  done "pyLib"

task buildCython, "Cython extension in-place":
  exec gate("pyLib")
  exec gate("pyDeps")
  exec "cd py && python3 setup.py build_ext --inplace"
  done "buildCython"

task pyTest, "Cython extension + pytest":
  exec gate("buildCython")
  exec "cd py && python3 -m pytest -q"
  done "pyTest"

task pyWheel, "wheel":
  exec gate("pyLib")
  exec gate("pyDeps")
  exec "cd py && python3 setup.py bdist_wheel"
  done "pyWheel"

task coverage, "LCOV + HTML coverage report for the Nim sources (needs lcov)":
  # gcov and lcov driven directly, no coco. Linux and macOS only.
  # --debugger:native attributes lines to the .nim sources, not the generated C.
  # --include keeps stdlib out of the capture, where lcov 2.x aborts on Nim's
  # codegen. Together they leave nothing to suppress: no --ignore-errors here,
  # so a real problem still fails the build.
  let cache = "build/covcache"
  rmDir cache
  rmDir "coverage"
  # Same cache dir across every test binary so gcov accumulates counters for
  # modules shared between suites (twoSum, algorithms/*) instead of losing
  # them to a fresh nimcache per compile.
  const covTests = [
    "test_eft", "test_naivesum", "test_pairwisesum", "test_compensatedsum",
    "test_shewchuksum", "test_exactsum", "test_orosum", "test_dotproduct",
    "test_expansions", "test_property"
  ]
  for t in covTests:
    exec "nim c --path:src --nimcache:" & cache &
         " --debugger:native --passC:--coverage --passL:--coverage" &
         " -o:build/cov_" & t & " tests/" & t & ".nim"
    exec "./build/cov_" & t
  # test_simd_dispatch needs -d:release, same as the dispatch task: debug-mode
  # NimContracts on naiveDot/dot2/dotK (compiled away in release) breaks the
  # {.raises: [].} on simd_dispatch.nim's scalar fallback wrappers.
  block:
    const t = "test_simd_dispatch"
    exec "nim c -d:release --path:src --nimcache:" & cache &
         " --debugger:native --passC:--coverage --passL:--coverage" &
         " -o:build/cov_" & t & " tests/" & t & ".nim"
    exec "./build/cov_" & t
  exec "lcov --capture --directory " & cache & " --base-directory ." &
       " --include \"*/src/UniAccurate/*\" --output-file lcov.info --quiet"
  exec "genhtml lcov.info --output-directory coverage --legend --quiet"
  exec "lcov --summary lcov.info"
  done "coverage"

# ---------------------------------------------------------------------------
# Full cross-backend comparator (bench/, ADR-0007): one column per backend,
# each writing bench/compare/{perf,cand}_<tag>.csv against the same canonical
# datasets, then `nimble benchAll` joins them against one correctly-rounded
# oracle. Heavier and slower than `nimble bench`'s smoke test -- run manually,
# never in CI.
# ---------------------------------------------------------------------------
task benchData, "Export the canonical bench datasets + manifest":
  exec "nim c -r --path:src bench/export_datasets.nim"
  done "benchData"

task benchNim, "UniAccurate scalar column (release)":
  exec "nim c -d:release --path:src -r bench/driver_nim.nim full nim"
  done "benchNim"

task benchNimFma, "UniAccurate scalar column with -d:useFMA (release)":
  exec "nim c -d:release -d:useFMA --path:src -r bench/driver_nim.nim full nim_fma"
  done "benchNimFma"

task benchNimSimd, "UniAccurate SIMD column with -d:simd (host ISA)":
  exec "nim c -d:release -d:simd" & simdArch &
       " --path:src -r bench/driver_nim.nim full nim_simd"
  done "benchNimSimd"

task benchC, "C originals column (FMA off)":
  exec "cc -O3 -ffp-contract=off -o bench/originals/driver_originals" &
       " bench/originals/driver_originals.c -lm"
  exec "./bench/originals/driver_originals"
  exec "mv -f bench/compare/perf_originals.csv bench/compare/perf_c.csv"
  exec "mv -f bench/compare/cand_originals.csv bench/compare/cand_c.csv"
  done "benchC"

task benchCFma, "C originals column with -mfma (FMA on)":
  exec "cc -O3 -ffp-contract=off -mfma -o bench/originals/driver_originals" &
       " bench/originals/driver_originals.c -lm"
  exec "./bench/originals/driver_originals"
  exec "mv -f bench/compare/perf_originals.csv bench/compare/perf_c_fma.csv"
  exec "mv -f bench/compare/cand_originals.csv bench/compare/cand_c_fma.csv"
  done "benchCFma"

task benchRust, "Rust accurate column (fma off, single-thread, needs cargo)":
  exec "cd bench/rust && cargo build --release --no-default-features"
  exec "./bench/rust/target/release/driver_rust"
  exec "mv -f bench/compare/perf_rust_raw.csv bench/compare/perf_rust.csv"
  exec "mv -f bench/compare/cand_rust_raw.csv bench/compare/cand_rust.csv"
  done "benchRust"

task benchRustFma, "Rust accurate column (fma on, single-thread, needs cargo)":
  exec "cd bench/rust && cargo build --release"
  exec "./bench/rust/target/release/driver_rust"
  exec "mv -f bench/compare/perf_rust_raw.csv bench/compare/perf_rust_fma.csv"
  exec "mv -f bench/compare/cand_rust_raw.csv bench/compare/cand_rust_fma.csv"
  done "benchRustFma"

task benchAll, "Aggregate every backend column into perf/acc matrices + summary.md":
  exec "nim c -r --path:src bench/aggregate.nim"
  done "benchAll"

task benchPython, "Python binding vs math.fsum/sum() (needs the built Cython extension)":
  exec gate("buildCython")
  exec "python3 py/bench_python.py"
  done "benchPython"

task benchReadme, "benchAll + benchPython, splice into README.md for this machine":
  exec gate("benchPython")
  exec "nim c -r --path:src bench/aggregate.nim --readme"
  done "benchReadme"
