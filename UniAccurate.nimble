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

task lint, "Fail if nimpretty would reformat a source":
  exec "nim c -r --hints:off -o:build/lint_tool tools/lint.nim"

task checkVGraph, "Fail on an import that climbs the layers in vgraph.cfg":
  exec "nim c -r --hints:off -o:build/vgraph_tool tools/vgraph.nim"

task docsDeps, "Install the docs toolchain (nimib)":
  exec "nimble install -y nimib"

task book, "Build the nimib book (needs nimib)":
  # nimib compiles and runs the book's code blocks: a drift fails the build.
  exec "nim c -r --path:src --hints:off -o:build/book book/index.nim"

task docs, "API reference + book into pages/ — what CI publishes":
  rmDir "pages"
  exec "nim doc --index:on --outdir:pages/api --project --hints:off src/UniAccurate.nim"
  exec "nimble book"
  # The book is the landing page; the generated reference sits under api/.
  cpFile "book/index.html", "pages/index.html"

task eft, "EFT tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_eft tests/test_eft.nim"

task sum, "Summation tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_naive tests/test_naivesum.nim"
  exec "nim c -r --path:src -o:build/test_pairwise tests/test_pairwisesum.nim"

task comp, "Compensated summation tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_comp tests/test_compensatedsum.nim"

task shewchuk, "Shewchuk summation tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_shewchuk tests/test_shewchuksum.nim"

task exact, "Exact superaccumulator tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_exact tests/test_exactsum.nim"

task oro, "ORO/Rump summation tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_oro tests/test_orosum.nim"

task dot, "Dot product tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_dot tests/test_dotproduct.nim"

task dispatch, "Runtime AVX2/AVX-512 dot dispatch tests (ADR-0008)":
  # -d:release, not debug: simd_dispatch.nim's {.raises: [].} on its scalar
  # fallback wrappers only holds once NimContracts' require/ensure on
  # naiveDot/dot2/dotK compile away (AGENTS.md) -- the real c_api.nim build is
  # always -d:release, so this test exercises the module the way it ships.
  exec "nim c -r -d:release --path:src -o:build/test_dispatch tests/test_simd_dispatch.nim"

task expansions, "Shewchuk expansion tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_expansions tests/test_expansions.nim"

task prop, "Randomized property tests (debug, contracts active)":
  exec "nim c -r --path:src -o:build/test_prop tests/test_property.nim"

task test, "Nim tests (debug, contracts active)":
  exec "nimble eft"
  exec "nimble sum"
  exec "nimble comp"
  exec "nimble shewchuk"
  exec "nimble exact"
  exec "nimble oro"
  exec "nimble dot"
  exec "nimble dispatch"
  exec "nimble expansions"
  exec "nimble prop"

task testRelease, "Nim tests (release, contracts compiled away)":
  exec "nim c -r -d:release --path:src -o:build/test_eft_rel tests/test_eft.nim"
  exec "nim c -r -d:release --path:src -o:build/test_naive_rel tests/test_naivesum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_pairwise_rel tests/test_pairwisesum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_comp_rel tests/test_compensatedsum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_shewchuk_rel tests/test_shewchuksum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_exact_rel tests/test_exactsum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_oro_rel tests/test_orosum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_dot_rel tests/test_dotproduct.nim"
  exec "nim c -r -d:release --path:src -o:build/test_dispatch_rel tests/test_simd_dispatch.nim"
  exec "nim c -r -d:release --path:src -o:build/test_expansions_rel tests/test_expansions.nim"
  exec "nim c -r -d:release --path:src -o:build/test_prop_rel tests/test_property.nim"

task testCi, "Nim tests (CI subset, debug)":
  exec "nimble eft"
  exec "nimble sum"
  exec "nimble comp"
  exec "nimble shewchuk"
  exec "nimble exact"
  exec "nimble oro"
  exec "nimble dot"
  exec "nimble dispatch"
  exec "nimble expansions"
  exec "nimble prop"

task testCiRelease, "Nim tests (CI subset, release)":
  exec "nim c -r -d:release --path:src -o:build/test_eft_rel tests/test_eft.nim"
  exec "nim c -r -d:release --path:src -o:build/test_naive_rel tests/test_naivesum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_pairwise_rel tests/test_pairwisesum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_comp_rel tests/test_compensatedsum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_shewchuk_rel tests/test_shewchuksum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_exact_rel tests/test_exactsum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_oro_rel tests/test_orosum.nim"
  exec "nim c -r -d:release --path:src -o:build/test_dot_rel tests/test_dotproduct.nim"
  exec "nim c -r -d:release --path:src -o:build/test_dispatch_rel tests/test_simd_dispatch.nim"
  exec "nim c -r -d:release --path:src -o:build/test_expansions_rel tests/test_expansions.nim"
  exec "nim c -r -d:release --path:src -o:build/test_prop_rel tests/test_property.nim"

task testAll, "debug + release + C ABI":
  exec "nimble test"
  exec "nimble testRelease"
  exec "nimble ctest"

task example, "Nim demo":
  exec "nim c -r --path:src -o:build/demo examples/demo.nim"

task bench, "Quick scalar bench smoke (release, reduced sizes) -- not the AVX-512 reference numbers":
  exec "nim c -r -d:release --path:src -o:build/bench_driver bench/driver_nim.nim quick ci"

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

task clibStatic, "C static library":
  exec "nim c --app:staticlib --noMain --mm:arc -d:release -o:" & staticLib &
       " src/UniAccurate/c_api.nim"

task clibMsvc, "C static library, MSVC ABI (Windows Python extension)":
  # CPython on Windows is MSVC-built and cannot link MinGW output.
  exec "nim c --cc:vcc --app:staticlib --noMain --mm:arc -d:release" &
       " -o:UniAccurate.lib src/UniAccurate/c_api.nim"

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
  exec "nimble clibStatic"
  exec makeExe & " -C tests/c" & makeWinArgs

task cexample, "C demo":
  exec "nimble clibStatic"
  exec makeExe & " -C examples/c" & makeWinArgsExample

task simd, "Run test_simd with -d:simd (host-default ISA)":
  exec "nim c -r -d:simd --path:src -o:build/test_simd tests/test_simd.nim"

task simdAvx2, "Run test_simd with AVX2 (amd64)":
  exec "nim c -r -d:simd -d:avx2 --path:src -o:build/test_simd_avx2 tests/test_simd.nim"

task simdAvx512, "Run test_simd with AVX-512 (amd64 Zen4)":
  exec "nim c -r -d:simd -d:avx512 --path:src -o:build/test_simd_avx512 tests/test_simd.nim"

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

task clibSimd, "C shared library with -d:simd (host ISA)":
  exec "nim c --app:lib --noMain --mm:arc -d:release -d:simd" & simdArch &
       " -o:" & sharedLib & macArgs & " src/UniAccurate/c_api.nim"

task ctestSimd, "C ABI tests with -d:simd (host ISA)":
  exec "nim c --app:staticlib --noMain --mm:arc -d:release -d:simd" & simdArch &
       " -o:" & staticLib & " src/UniAccurate/c_api.nim"
  exec makeExe & " -C tests/c" & makeWinArgs

task pyDeps, "Install Python build deps (setuptools, Cython, pytest) if missing":
  exec "python3 -m pip install --break-system-packages --quiet setuptools wheel \"Cython>=3.0.0\" pytest"

# The extension links the vcc static lib on Windows, the shared lib elsewhere.
task pyLib, "Build the library the Python extension links against":
  when defined(windows):
    exec "nimble clibMsvc"
  else:
    exec "nimble clib"

task buildCython, "Cython extension in-place":
  exec "nimble pyLib"
  exec "nimble pyDeps"
  exec "cd py && python3 setup.py build_ext --inplace"

task pyTest, "Cython extension + pytest":
  exec "nimble buildCython"
  exec "cd py && python3 -m pytest -q"

task pyWheel, "wheel":
  exec "nimble pyLib"
  exec "nimble pyDeps"
  exec "cd py && python3 setup.py bdist_wheel"

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

# ---------------------------------------------------------------------------
# Full cross-backend comparator (bench/, ADR-0007): one column per backend,
# each writing bench/compare/{perf,cand}_<tag>.csv against the same canonical
# datasets, then `nimble benchAll` joins them against one correctly-rounded
# oracle. Heavier and slower than `nimble bench`'s smoke test -- run manually,
# never in CI.
# ---------------------------------------------------------------------------
task benchData, "Export the canonical bench datasets + manifest":
  exec "nim c -r --path:src bench/export_datasets.nim"

task benchNim, "UniAccurate scalar column (release)":
  exec "nim c -d:release --path:src -r bench/driver_nim.nim full nim"

task benchNimFma, "UniAccurate scalar column with -d:useFMA (release)":
  exec "nim c -d:release -d:useFMA --path:src -r bench/driver_nim.nim full nim_fma"

task benchNimSimd, "UniAccurate SIMD column with -d:simd (host ISA)":
  exec "nim c -d:release -d:simd" & simdArch &
       " --path:src -r bench/driver_nim.nim full nim_simd"

task benchC, "C originals column (FMA off)":
  exec "cc -O3 -ffp-contract=off -o bench/originals/driver_originals" &
       " bench/originals/driver_originals.c -lm"
  exec "./bench/originals/driver_originals"
  exec "mv -f bench/compare/perf_originals.csv bench/compare/perf_c.csv"
  exec "mv -f bench/compare/cand_originals.csv bench/compare/cand_c.csv"

task benchCFma, "C originals column with -mfma (FMA on)":
  exec "cc -O3 -ffp-contract=off -mfma -o bench/originals/driver_originals" &
       " bench/originals/driver_originals.c -lm"
  exec "./bench/originals/driver_originals"
  exec "mv -f bench/compare/perf_originals.csv bench/compare/perf_c_fma.csv"
  exec "mv -f bench/compare/cand_originals.csv bench/compare/cand_c_fma.csv"

task benchRust, "Rust accurate column (fma off, single-thread, needs cargo)":
  exec "cd bench/rust && cargo build --release --no-default-features"
  exec "./bench/rust/target/release/driver_rust"
  exec "mv -f bench/compare/perf_rust_raw.csv bench/compare/perf_rust.csv"
  exec "mv -f bench/compare/cand_rust_raw.csv bench/compare/cand_rust.csv"

task benchRustFma, "Rust accurate column (fma on, single-thread, needs cargo)":
  exec "cd bench/rust && cargo build --release"
  exec "./bench/rust/target/release/driver_rust"
  exec "mv -f bench/compare/perf_rust_raw.csv bench/compare/perf_rust_fma.csv"
  exec "mv -f bench/compare/cand_rust_raw.csv bench/compare/cand_rust_fma.csv"

task benchAll, "Aggregate every backend column into perf/acc matrices + summary.md":
  exec "nim c -r --path:src bench/aggregate.nim"

task benchPython, "Python binding vs math.fsum/sum() (needs the built Cython extension)":
  exec "nimble buildCython"
  exec "python3 py/bench_python.py"

task benchReadme, "benchAll + benchPython, splice into README.md for this machine":
  exec "nimble benchPython"
  exec "nim c -r --path:src bench/aggregate.nim --readme"
