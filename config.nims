# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## UniAccurate build config.
##
## FMA lowering. Two separate concerns:
##
##   1. `-ffp-contract=off` (correctness) — the C compiler must never rewrite
##      `a*b - c` into a fused `fma(a,b,-c)`. That contraction silently breaks
##      the Dekker error identity of `twoProduct` (the `ah*bh - result[0]`
##      term must round separately). clang contracts by default on both amd64
##      and arm64, so the guard is applied on both. This is the FMA contract
##      invariant (ADR-0004). GCC does not contract by default and MSVC's
##      `/fp:precise` already forbids it (and would not understand the flag),
##      so the switch is GCC/Clang-only.
##
##   2. `-mfma` (amd64, opt-in) — lets the compiler inline the C99 `fma`/`fmaf`
##      used by `twoProductFMA` to a single hardware instruction, but it assumes
##      the *target CPU* has FMA: a generic wheel run on an older amd64 CPU
##      without FMA would SIGILL. Default off — libm `fma` stays a call the
##      platform dispatches (glibc IFUNC, arm64 base-ISA FMA), portable and
##      still fast. Pass `-d:useFMA` for a CPU-specific build that assumes FMA.
##      arm64 has FMA in the ISA, so no flag there.
##
## Opt out of both with `-d:scalarUniAccurate`.
when not defined(scalarUniAccurate):
  when defined(gcc) or defined(clang):
    switch("passC", "-ffp-contract=off")
    when defined(amd64) and defined(useFMA):
      switch("passC", "-mfma")

## SIMD target flags: `-d:simd` pulls the simd layer; `-d:avx2`/`-d:avx512`
## select the amd64 ISA (nimsimd branches on them). arm64 NEON is base ISA.
when defined(simd) and not defined(scalarUniAccurate):
  when defined(amd64):
    # The SIMD dot kernels use FMA intrinsics (mm{256,512}_fmadd_pd, ADR-0007 Lever 1),
    # so the amd64 target flags must enable FMA too. -mfma is portable alongside
    # -mavx2/-mavx512f: every AVX2 CPU (Haswell+) has FMA. arm64 NEON has FMA in
    # the base ISA, so no flag there.
    when defined(avx512):
      when defined(vcc):
        switch("passC", "/arch:AVX512")
      else:
        switch("passC", "-mavx512f -mfma")
    elif defined(avx2):
      when defined(vcc):
        switch("passC", "/arch:AVX2")
      else:
        switch("passC", "-mavx2 -mfma")
