## UniAccurate build config.
##
## FMA lowering. Two separate concerns:
##
##   1. `-ffp-contract=off` (correctness) — the C compiler must never rewrite
##      `a*b - c` into a fused `fma(a,b,-c)`. That contraction silently breaks
##      the Dekker error identity of `twoProduct` (the `ah*bh - result[0]`
##      term must round separately). clang contracts by default on both amd64
##      and arm64, so the guard is applied on both. This is the FMA contract
##      invariant (ADR-0005).
##
##   2. `-mfma` (amd64 perf only) — enables the FMA instruction set so the C99
##      `fma`/`fmaf` intrinsics used by `twoProductFMA` map to a single hardware
##      instruction. arm64 ships with FMA in the ISA, so no flag there.
##
## Opt out of both with `-d:scalarUniAccurate`.
when not defined(scalarUniAccurate):
  when defined(amd64):
    switch("passC", "-mfma")
    switch("passC", "-ffp-contract=off")
  elif defined(arm64):
    switch("passC", "-ffp-contract=off")
