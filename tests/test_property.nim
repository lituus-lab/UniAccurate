# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import std/unittest
import std/math
import UniAccurate

## Randomized property tests for the summation algorithms. Oracle-free
## invariants that hold for every input: finite inputs never yield NaN (even
## at scale and through overflow), and every sum agrees exactly on
## integer-valued data (within 2^53). Rigorous forward-bound checks live in
## `py/tests/test_oracle.py`; this file complements them with scale and float32
## coverage on the native (non-C-ABI) call path.

proc next(r: var uint64): uint64 =
  ## Deterministic xorshift64 so every run is reproducible (no RNG state).
  var x = r
  x = x xor (x shl 13)
  x = x xor (x shr 7)
  x = x xor (x shl 17)
  r = x
  x

proc randomF64(r: var uint64): float64 =
  ## Finite normal float64, exponent in [-511, 510]: no under/overflow, no NaN.
  let expField = uint64(next(r) mod 1022 + 512)
  var bits = expField shl 52
  bits = bits or (next(r) and 0x000F_FFFF_FFFF_FFFF'u64)
  cast[float64](bits)

proc randomSignedBigF64(r: var uint64): float64 =
  ## float64 near the top of the range (exp in [1017, 1023]) with a random
  ## sign. Summing many overflows; mixed signs let a subtree hit +Inf or -Inf.
  let expField = uint64(next(r) mod 7 + 2040)
  var bits = expField shl 52
  bits = bits or (next(r) and 0x000F_FFFF_FFFF_FFFF'u64)
  if (next(r) and 1) == 1:
    bits = bits or (1'u64 shl 63) # negative
  cast[float64](bits)

proc randomF32(r: var uint64): float32 =
  ## Finite normal float32, exponent in [-62, 63]: no under/overflow, no NaN.
  let expField = uint32(next(r) mod 126 + 65)
  var bits = expField shl 23
  bits = bits or (uint32(next(r)) and 0x007F_FFFF'u32)
  cast[float32](bits)

proc randomIntF64(r: var uint64): float64 =
  ## Integer-valued float64 in [-1e6, 1e6]; exact, and sums of up to ~1e4 of
  ## them stay well below 2^53.
  let v = int64(next(r) mod 2_000_001) - 1_000_000
  float64(v)

const Seed = 0x9E37_79B9_7F4A_7C15'u64

suite "finite inputs never yield NaN (no overflow)":
  var r = Seed
  test "all sums, bounded random float64 up to n=100000":
    for n in [1, 2, 3, 10, 100, 1000, 10000, 100000]:
      var x = newSeq[float64](n)
      for i in 0 ..< n:
        x[i] = randomF64(r)
      check classify(naiveSum(x)) != fcNan
      check classify(pairwiseSum(x)) != fcNan
      check classify(pairwiseSumIterative(x)) != fcNan
      check classify(kahanSum(x)) != fcNan
      check classify(neumaierSum(x)) != fcNan
      check classify(kleinSum(x)) != fcNan
      check classify(shewchukSum(x)) != fcNan
      check classify(superSum(x)) != fcNan
      check classify(sum2(x)) != fcNan
      check classify(sumK(x, 2)) != fcNan
      check classify(accSum(x)) != fcNan
      check classify(nearSum(x)) != fcNan

suite "finite inputs never yield NaN (overflow-prone)":
  # Naive and the compensated sums carry the per-step `isFin` guard (or, for
  # naive, a single running accumulator that cannot flip sign once Inf), so
  # finite inputs never evaluate Inf - Inf. Pairwise is excluded here: its
  # merge tree can hit +Inf and -Inf in two subtrees and yield NaN — a
  # documented limitation.
  var r = Seed xor 1
  test "naive and compensated, signed big float64 up to n=10000":
    for n in [1, 10, 100, 1000, 10000]:
      var x = newSeq[float64](n)
      for i in 0 ..< n:
        x[i] = randomSignedBigF64(r)
      check classify(naiveSum(x)) != fcNan
      check classify(kahanSum(x)) != fcNan
      check classify(neumaierSum(x)) != fcNan
      check classify(kleinSum(x)) != fcNan
      check classify(shewchukSum(x)) != fcNan
      check classify(superSum(x)) != fcNan
      # The Rump family falls back to `superSum` on non-finite/overflow input,
      # so it inherits the same finite-never-NaN property (correctly rounded,
      # stronger than faithful).
      check classify(sum2(x)) != fcNan
      check classify(sumK(x, 2)) != fcNan
      check classify(accSum(x)) != fcNan
      check classify(nearSum(x)) != fcNan

suite "exact integer data agrees across all sums":
  var r = Seed xor 2
  test "naive == pairwise == iterative == kahan == neumaier == klein == shewchuk == super":
    for n in [1, 2, 3, 10, 100, 500]:
      var x = newSeq[float64](n)
      for i in 0 ..< n:
        x[i] = randomIntF64(r)
      let want = naiveSum(x)
      check pairwiseSum(x) == want
      check pairwiseSumIterative(x) == want
      check kahanSum(x) == want
      check neumaierSum(x) == want
      check kleinSum(x) == want
      check shewchukSum(x) == want
      check superSum(x) == want
      # Integer-valued data sums exactly at every precision, so the ORO/Rump
      # family (compensated, faithful, correctly rounded) all hit the exact
      # integer — `sumK` at K=1..3, `accSum` faithful, `nearSum` rounded.
      check sum2(x) == want
      check sumK(x, 1) == want
      check sumK(x, 2) == want
      check sumK(x, 3) == want
      check accSum(x) == want
      check nearSum(x) == want

suite "float32 compensated finite never NaN":
  var r = Seed xor 3
  test "kahan/neumaier/klein/shewchuk/super, bounded random float32 up to n=10000":
    for n in [1, 10, 100, 1000, 10000]:
      var x = newSeq[float32](n)
      for i in 0 ..< n:
        x[i] = randomF32(r)
      check classify(kahanSum(x)) != fcNan
      check classify(neumaierSum(x)) != fcNan
      check classify(kleinSum(x)) != fcNan
      check classify(shewchukSum(x)) != fcNan
      check classify(superSum(x)) != fcNan
      check classify(sum2(x)) != fcNan
      check classify(sumK(x, 2)) != fcNan
      check classify(accSum(x)) != fcNan
      check classify(nearSum(x)) != fcNan
