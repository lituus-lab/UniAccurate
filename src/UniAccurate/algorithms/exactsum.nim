# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Correctly-rounded summation and dot product via a small superaccumulator.
##
## `superSum` / `superDot` accumulate every term exactly into a fixed-memory
## integer superaccumulator and round once to nearest (IEEE-754 ties-to-even).
## The result is independent of the number of summands and of the condition
## number, and equals `math.fsum` bit-for-bit for finite, non-overflowing
## float64 input.
##
## Algorithm
## =========
##
## Radford Neal's **small superaccumulator**: the value is held as signed 64-bit
## chunks (two's-complement) whose union is the exact integer sum. Adding a
## float decomposes its exponent (high bits → chunk index, low bits → bit
## offset within the chunk) and its mantissa (a low part added to the chunk, a
## high part added to the next chunk up), branchlessly via a two's-complement
## conditional-negation trick. Carries between chunks are propagated lazily,
## only often enough that they cannot overflow a chunk. The final `round` finds
## the leading bit, pulls guard and sticky bits from lower chunks, and rounds
## to nearest-even.
##
## The per-term add is a **branchless integer update**, self-contained (pure
## integer bit ops, no error-free-transform chain), which keeps the hot path
## amenable to SIMD.
##
## This is the **small** accumulator, sized to cover the full *product* exponent
## range (not just sums): 131 chunks for float64, 35 for float32 —
## `(1 shl highExpBits) + 3` with `highExpBits = 7` (float64) / `5` (float32).
## A sum of float64 values reaches biased exponent 2046 → chunk 63, but a
## *product* of two max-magnitude values reaches 2^2046 → chunk ~97, so the
## accumulator is widened past the sum range to hold exact dot products
## (`addProduct` / `superDot`). Exact for any number of terms. The **large**
## accumulator (a 4096-chunk lazy layer for very large streams) and the
## SSE/AVX intrinsics are out of scope.
##
## Only `+`, integer bit ops, and one `countLeadingZeroBits` (in `round`, to
## locate the leading bit — replacing an int→float conversion so the same code
## serves float32 and float64) are used.
##
## Correctness assumptions (IEEE-754)
## ==================================
##
## The accumulation is exact integer arithmetic on the IEEE-754 binary{32,64}
## bit patterns, so `FLT_EVAL_METHOD` and `-ffast-math` (which govern
## floating-point operation order and precision) do not apply — there is no
## mid-stream FP operation to reorder or extend. The result is determined by
## the input format and the final `round`, built from integer guard/sticky
## bits to honor roundTiesToEven. Under those, the correctly-rounded guarantee
## is order-independent and exact.
##
## Contracts
## =========
##
## `{.contractual.}` on `superSum` / `superDot` (NimContracts, debug-only,
## compiled away under `-d:release`). Postconditions: `[]` sums to zero, and
## finite input ⇒ the result is never NaN (overflow ⇒ ±Inf). `superDot` adds
## the length-matching precondition `x.len == y.len`. The finite-input safety
## holds because the exact integer accumulator has no `Inf − Inf` order
## artifact: overflow gives correctly-signed ±Inf, cancellation gives the true
## finite value. The correctly-rounded guarantee itself is real-arithmetic,
## verified against a `math.fsum` oracle in `tests/test_exactsum.nim`, not by a
## float-level contract.
##
## References
## ==========
##
## - Neal, R.M. (2015). "Fast Exact Summation Using Small and Large
##   Superaccumulators." arXiv:1505.05571. doi:10.48550/arXiv.1505.05571.
## - Zhu, Y.K. & Hayes, W.B. (2010). "Algorithm 908: Online Exact Summation of
##   Floating-Point Streams." *ACM Trans. Math. Softw.* 37(3).
##   doi:10.1145/1824801.1824815 — an alternative exact online method, not
##   implemented here.
import std/bitops
import std/math
import contracts
when not defined(release) and not defined(danger):
  import ../twosum # allFin: finite-input safety postcondition

# ----------------------------------------------------------------------------
# Bit-pattern helpers (the accumulator stores the exact integer sum; Inf/NaN
# are tracked by their IEEE bit patterns so payload comparison is exact).
# ----------------------------------------------------------------------------

func toBits[T: SomeFloat](v: T): uint64 {.inline.} =
  when T is float64: cast[uint64](v) else: cast[uint32](v).uint64

func fromBits[T: SomeFloat](b: uint64): T {.inline.} =
  when T is float64: cast[T](b) else: cast[T](uint32(b))

# ----------------------------------------------------------------------------
# The small superaccumulator
# ----------------------------------------------------------------------------

type SuperAccumulator*[T: SomeFloat] = object
  ## Neal's small superaccumulator: the exact integer sum held as signed 64-bit
  ## chunks (131 for float64, 35 for float32 — `(1 shl highExpBits) + 3`,
  ## widened to cover the product exponent range), plus pending Inf/NaN bit
  ## patterns and a lazy carry counter. Online and fixed-memory; `round`
  ## produces the correctly-rounded `T`.
  chunk: array[when T is float64: 131 else: 35, int64]
  inf: T ## Pending ±Inf (or NaN from +Inf − −Inf); `T(0)` if none.
  nan: T ## Pending NaN (largest payload wins); `T(0)` if none.
  addsUntilPropagate: int ## Adds left before carries must be propagated.

func initSuperAccumulator*[T: SomeFloat](acc: var SuperAccumulator[T]) {.
    inline.} =
  ## Zero `acc` and reset its carry counter. Required before any `add`.
  when T is float64:
    const CarryTerms = (1 shl 11) - 1
  else:
    const CarryTerms = (1 shl 40) - 1
  for i in 0 ..< acc.chunk.len:
    acc.chunk[i] = 0
  acc.inf = T(0)
  acc.nan = T(0)
  acc.addsUntilPropagate = CarryTerms

func addInfNan[T: SomeFloat](acc: var SuperAccumulator[T], value: T) {.
    inline.} =
  ## Record an Inf or NaN per Neal's `xsum_small_add_inf_nan`: a second Inf of
  ## opposite sign produces NaN; among NaNs the one with the larger payload
  ## wins (sign cleared).
  when T is float64:
    const MantBits = 52
    const SignBit = uint64(1) shl 63
  else:
    const MantBits = 23
    const SignBit = uint64(1) shl 31
  const MantMask = (uint64(1) shl MantBits) - 1
  let uv = toBits(value)
  if (uv and MantMask) == 0: # Inf
    if acc.inf == T(0):
      acc.inf = value
    elif toBits(acc.inf) != uv: # opposite-sign Inf → NaN
      acc.inf = value - value
  else: # NaN: bigger payload wins, sign cleared
    if (toBits(acc.nan) and MantMask) <= (uv and MantMask):
      acc.nan = fromBits[T](uv and not SignBit)

func add1NoCarry[T: SomeFloat](acc: var SuperAccumulator[T], value: T) {.
    inline.} =
  ## Add one value to the exact integer accumulator, assuming no carry
  ## propagation is due (Neal's `xsum_add1_no_carry`). Branchless: the
  ## mantissa is split across two consecutive chunks and added or subtracted
  ## via a two's-complement sign mask.
  when T is float64:
    const MantBits = 52
    const ExpMask = 0x7FF
    const LowExpBits = 5
    const LowMantBits = 32
  else:
    const MantBits = 23
    const ExpMask = 0xFF
    const LowExpBits = 4
    const LowMantBits = 16
  const LowMantMask = (int64(1) shl LowMantBits) - 1

  # Bit pattern in the low bits (float32: 32 bits, zero-extended; float64: 64
  # bits) for exponent/mantissa extraction, and a sign-extended int64 so the
  # sign lands at bit 63 and the branchless `ashr(_, 63)` trick works for both.
  let uv = toBits(value)
  let iv = when T is float64: cast[int64](uv)
           else: cast[int64](cast[int32](uint32(uv)))
  var exp = int((uv shr MantBits) and uint64(ExpMask))
  var mant = uv and ((uint64(1) shl MantBits) - 1)
  var lowExp = exp and ((1 shl LowExpBits) - 1)
  let highExp = exp shr LowExpBits

  if exp == 0: # zero or denormalized
    if mant == 0:
      return # ±0 contributes nothing
    exp = 1 # denormalized: exponent treated as 1
    lowExp = 1
  elif exp == ExpMask: # Inf or NaN
    addInfNan(acc, value)
    return
  else: # normalized: set the implicit leading 1
    mant = mant or (uint64(1) shl MantBits)

  # Split the mantissa (shifted by lowExp) into a low part (≤ LowMantBits,
  # added to chunk[highExp]) and a high part (added to chunk[highExp+1], which
  # always exists — there are three spare chunks at the top).
  let lowMant = int64((mant shl lowExp) and uint64(LowMantMask))
  let highMant = int64(mant shr (LowMantBits - lowExp))

  # Branchless signed add: neg is 0 for non-negative, -1 (all ones) for
  # negative; (x xor neg) + (neg and 1) is x or -x.
  let neg = ashr(iv, 63)
  acc.chunk[highExp] += (lowMant xor neg) + (neg and 1)
  acc.chunk[highExp + 1] += (highMant xor neg) + (neg and 1)

func carryPropagate[T: SomeFloat](acc: var SuperAccumulator[T]): int =
  ## Propagate carries between chunks (Neal's `xsum_carry_propagate`, scalar
  ## path) so the uppermost non-zero chunk carries the sign and no chunk holds
  ## more than `LowMantBits` bits. Returns the index of that uppermost non-zero
  ## chunk (0 if the sum is zero), and resets the lazy carry counter.
  when T is float64:
    const LowMantBits = 32
    const Schunks = 131
    const MantBits = 52
    const ExpMask = 0x7FF
  else:
    const LowMantBits = 16
    const Schunks = 35
    const MantBits = 23
    const ExpMask = 0xFF
  const LowMantMask = (int64(1) shl LowMantBits) - 1
  when T is float64:
    const CarryTerms = (1 shl 11) - 1
  else:
    const CarryTerms = (1 shl 40) - 1

  # Uppermost non-zero chunk (scan from the top).
  var u = Schunks - 1
  while u > 0 and acc.chunk[u] == 0:
    dec u
  if acc.chunk[u] == 0: # all zero
    acc.addsUntilPropagate = CarryTerms - 1
    return 0

  var i = 0
  var uix = -1 # uppermost non-zero chunk after propagation
  while i <= u:
    let c = acc.chunk[i]
    if c == 0:
      inc i
      continue
    let chigh = ashr(c, LowMantBits)
    if chigh == 0: # this chunk already fits
      uix = i
      inc i
      continue
    if u == i:
      if chigh == -1: # sign-extension: do not push -1 into zeros
        uix = i
        break
      u = i + 1 # chunk[i+1] will change; extend the sweep
    let clow = c and LowMantMask
    if clow != 0:
      uix = i
    acc.chunk[i] = clow
    if i + 1 >= Schunks: # overflow past the top → NaN
      addInfNan(acc, fromBits[T](uint64(ExpMask) shl MantBits or
                                 ((uint64(1) shl MantBits) - 1)))
      u = i
    else:
      acc.chunk[i + 1] += chigh # may zero chunk[i+1], hence the rescan logic
    inc i

  if uix < 0: # carry propagation cancelled everything
    acc.addsUntilPropagate = CarryTerms - 1
    return 0

  # Absorb a trailing -1 top chunk into the chunk below (a borrow), keeping the
  # sign in the lowest non-zero chunk.
  while acc.chunk[uix] == -1 and uix > 0:
    acc.chunk[uix - 1] += int64(-1) * (int64(1) shl LowMantBits)
    acc.chunk[uix] = 0
    dec uix

  acc.addsUntilPropagate = CarryTerms - 1
  return uix

func add*[T: SomeFloat](acc: var SuperAccumulator[T], value: T) {.inline.} =
  ## Add one value to `acc`, propagating carries lazily when the counter runs
  ## out (Neal's `xsum_small_add1`). Accepts any `T`, including NaN/±Inf.
  if acc.addsUntilPropagate == 0:
    discard carryPropagate(acc)
  add1NoCarry(acc, value)
  dec acc.addsUntilPropagate

func add*[T: SomeFloat](acc: var SuperAccumulator[T], values: openArray[T]) =
  ## Add a vector of values to `acc` (Neal's `xsum_small_addv`): a tight
  ## `add1NoCarry` loop with carries propagated only when the counter exhausts.
  ## This branchless inner loop is the hot path of the exact sum.
  var n = values.len
  var k = 0
  while n > 0:
    if acc.addsUntilPropagate == 0:
      discard carryPropagate(acc)
    let m = if n <= acc.addsUntilPropagate: n else: acc.addsUntilPropagate
    for j in 0 ..< m:
      add1NoCarry(acc, values[k + j])
    acc.addsUntilPropagate -= m
    k += m
    n -= m

func addProduct*[T: SomeFloat](acc: var SuperAccumulator[T], x, y: T) {.
    inline.} =
  ## Add the *exact* product `x·y` to `acc` without first rounding it to `T` —
  ## the fused multiply-accumulate that `superDot` builds on. Holds the full
  ## product significand (106-bit for float64, 48-bit for float32), so a
  ## product that overflows `T` (e.g. `maxValue·maxValue ≈ 2^2046`, far beyond
  ## the float64 range) is accumulated at its true magnitude rather than
  ## collapsed to ±Inf. The combined biased exponent `F = ex + ey − 1075`
  ## (float64) / `− 150` (float32) maps to chunk `F shr LowExpBits`; the
  ## significand is scattered across the chunks it spans (up to 5 for float64,
  ## 4 for float32) with the same branchless signed-add trick as `add1NoCarry`.
  ## Slices that fall below chunk 0 (sub-representable, only sub-ulp sticky)
  ## are dropped, matching the accumulator's inherent low-end bound; this never
  ## produces NaN. NaN/±Inf operands delegate to the IEEE product `x·y` (always
  ## Inf/NaN) and `addInfNan`. Carry propagation is lazy, as in `add(value)`.
  if acc.addsUntilPropagate == 0:
    discard carryPropagate(acc)
  when T is float64:
    const MantBits = 52
    const ExpMask = 0x7FF
    const LowExpBits = 5
    const SignBit = uint64(1) shl 63
    const MantMask = (uint64(1) shl MantBits) - 1
    const Mask32 = (uint64(1) shl 32) - 1
    let ux = toBits(x)
    let uy = toBits(y)
    let expx = int((ux shr MantBits) and uint64(ExpMask))
    let expy = int((uy shr MantBits) and uint64(ExpMask))
    if expx == ExpMask or expy == ExpMask: # Inf/NaN operand → IEEE product
      addInfNan(acc, x * y)
      return
    let mantx0 = ux and MantMask
    let manty0 = uy and MantMask
    if (expx == 0 and mantx0 == 0) or (expy == 0 and manty0 == 0): # ±0 → nothing
      return
    # 53-bit significands; denormals use effective biased exponent 1.
    var mx = mantx0
    var ex = expx
    if ex == 0:
      ex = 1
    else:
      mx = mx or (uint64(1) shl MantBits)
    var my = manty0
    var ey = expy
    if ey == 0:
      ey = 1
    else:
      my = my or (uint64(1) shl MantBits)
    let neg = if ((ux xor uy) and SignBit) != 0: -1'i64 else: 0'i64
    # 106-bit product P = (hi:lo) via 64×64→128 (32-bit half-word split).
    let ax = mx shr 32
    let bx = mx and Mask32
    let ay = my shr 32
    let by = my and Mask32
    let ll = bx * by # < 2^64
    let lh = bx * ay # < 2^53
    let hl = ax * by # < 2^53
    let hh = ax * ay # < 2^42
    let mid = lh + hl # < 2^54
    let midLo = mid and Mask32
    let midHi = mid shr 32 # < 2^22
    var plo = ll + (midLo shl 32)
    var phi = hh + midHi
    if plo < ll:
      phi += 1'u64 # carry from ll + midLo·2^32
    # Scatter Q = P shl lowExp (a 3-limb 137-bit value) across five 32-bit chunks.
    let f = ex + ey - 1075
    let highExp = f shr LowExpBits
    let lowExp = f and ((1 shl LowExpBits) - 1)
    let q0 = plo shl lowExp
    let q1 = if lowExp == 0: phi else: (phi shl lowExp) or (plo shr (64 - lowExp))
    let q2 = if lowExp == 0: 0'u64 else: phi shr (64 - lowExp)
    template put(k: int, sv: int64) =
      let idx = highExp + k
      if idx >= 0 and idx < acc.chunk.len:
        acc.chunk[idx] += (sv xor neg) + (neg and 1)
    put(0, int64(q0 and Mask32))
    put(1, int64(q0 shr 32))
    put(2, int64(q1 and Mask32))
    put(3, int64(q1 shr 32))
    put(4, int64(q2 and Mask32))
  else: # float32: 24-bit significands → 48-bit product fits a uint64.
    const MantBits = 23
    const ExpMask = 0xFF
    const LowExpBits = 4
    const SignBit = uint64(1) shl 31
    const MantMask = (uint64(1) shl MantBits) - 1
    const Mask16 = (uint64(1) shl 16) - 1
    let ux = toBits(x)
    let uy = toBits(y)
    let expx = int((ux shr MantBits) and uint64(ExpMask))
    let expy = int((uy shr MantBits) and uint64(ExpMask))
    if expx == ExpMask or expy == ExpMask:
      addInfNan(acc, x * y)
      return
    let mantx0 = ux and MantMask
    let manty0 = uy and MantMask
    if (expx == 0 and mantx0 == 0) or (expy == 0 and manty0 == 0):
      return
    var mx = mantx0
    var ex = expx
    if ex == 0:
      ex = 1
    else:
      mx = mx or (uint64(1) shl MantBits)
    var my = manty0
    var ey = expy
    if ey == 0:
      ey = 1
    else:
      my = my or (uint64(1) shl MantBits)
    let neg = if ((ux xor uy) and SignBit) != 0: -1'i64 else: 0'i64
    let p = mx * my # < 2^48, exact
    let f = ex + ey - 150
    let highExp = f shr LowExpBits
    let lowExp = f and ((1 shl LowExpBits) - 1)
    let q = p shl lowExp # < 2^64 (48 + 16)
    template put(k: int, sv: int64) =
      let idx = highExp + k
      if idx >= 0 and idx < acc.chunk.len:
        acc.chunk[idx] += (sv xor neg) + (neg and 1)
    put(0, int64(q and Mask16))
    put(1, int64((q shr 16) and Mask16))
    put(2, int64((q shr 32) and Mask16))
    put(3, int64(q shr 48))
  dec acc.addsUntilPropagate

func round*[T: SomeFloat](acc: var SuperAccumulator[T]): T =
  ## Produce the correctly-rounded `T` from `acc` (Neal's `xsum_small_round`):
  ## return a pending NaN/Inf; otherwise propagate carries, handle zero and
  ## denormals directly, and for normals locate the leading bit, gather guard
  ## and sticky bits, and round to nearest-even. Overflow yields ±Inf.
  when T is float64:
    const MantBits = 52
    const ExpMask = 0x7FF
    const ExpBias = 1023
    const LowExpBits = 5
    const LowMantBits = 32
    const SignMask = uint64(1) shl 63
  else:
    const MantBits = 23
    const ExpMask = 0xFF
    const ExpBias = 127
    const LowExpBits = 4
    const LowMantBits = 16
    const SignMask = uint64(1) shl 31
  const MantMask = (uint64(1) shl MantBits) - 1

  if acc.nan != T(0):
    return acc.nan
  if acc.inf != T(0):
    return acc.inf

  let i = carryPropagate(acc)
  var ivalue = acc.chunk[i]

  # Zero and denormalized results (only the lowest one or two chunks).
  if i <= 1:
    if ivalue == 0:
      return T(0)
    if i == 0:
      var intv = if ivalue >= 0: uint64(ivalue) else: uint64(-ivalue)
      intv = intv shr 1
      if ivalue < 0:
        intv = intv or SignMask
      return fromBits[T](intv)
    else: # i == 1
      var intv = ivalue * (int64(1) shl (LowMantBits - 1)) +
                 ashr(acc.chunk[0], 1)
      if intv < 0:
        if intv > -(int64(1) shl MantBits):
          return fromBits[T](uint64(-intv) or SignMask)
      else:
        if uint64(intv) < (uint64(1) shl MantBits):
          return fromBits[T](uint64(intv))
      # otherwise not actually denormalized; fall through to the normal path

    # Locate the leading bit of |ivalue| via countLeadingZeroBits (replaces an
    # int→float conversion, so the same code serves float32 and float64).
  let absv = if ivalue >= 0: uint64(ivalue) else: uint64(-ivalue)
  let e = int(63 - countLeadingZeroBits(absv)) + ExpBias
  var more = 2 + MantBits + ExpBias - e

  # Pull `more` bits from lower chunks into ivalue; keep the leftover `lower`.
  ivalue = ivalue * (int64(1) shl more)
  var j = i - 1
  var lower = if j < 0: int64(0) else: acc.chunk[j]
  if more >= LowMantBits:
    more -= LowMantBits
    ivalue += lower shl more
    dec j
    lower = if j < 0: int64(0) else: acc.chunk[j]
  ivalue += ashr(lower, LowMantBits - more)
  lower = lower and ((int64(1) shl (LowMantBits - more)) - 1)

  var intv: uint64
  var eFinal = e

  if ivalue >= 0: # positive: lower bits add to the magnitude
    intv = 0
    if (ivalue and 2) == 0: # extra bits 0x: remainder < 1/2
      discard
    elif (ivalue and 1) != 0: # 11: remainder > 1/2
      ivalue += 4
      if (ivalue and (int64(1) shl (MantBits + 3))) != 0:
        ivalue = ivalue shr 1
        inc eFinal
    elif (ivalue and 4) != 0: # 10, mantissa low bit 1 (odd): round away
      ivalue += 4
      if (ivalue and (int64(1) shl (MantBits + 3))) != 0:
        ivalue = ivalue shr 1
        inc eFinal
    else: # 10, even: round away only if sticky is set
      var sticky = lower
      if sticky == 0:
        while j > 0:
          dec j
          if acc.chunk[j] != 0:
            sticky = 1
            break
      if sticky != 0:
        ivalue += 4
        if (ivalue and (int64(1) shl (MantBits + 3))) != 0:
          ivalue = ivalue shr 1
          inc eFinal
  else: # negative: lower bits subtract from magnitude
    # If negating would lose the top bit, pull one more bit from `lower`.
    if ((-ivalue) and (int64(1) shl (MantBits + 2))) == 0:
      let pos = int64(1) shl (LowMantBits - 1 - more)
      ivalue = ivalue * 2
      if (lower and pos) != 0:
        inc ivalue
        lower = lower and not pos
      dec eFinal
    intv = SignMask
    ivalue = -ivalue # ivalue now holds the absolute value
    if (ivalue and 3) == 3: # 11: round away
      ivalue += 4
      if (ivalue and (int64(1) shl (MantBits + 3))) != 0:
        ivalue = ivalue shr 1
        inc eFinal
    elif (ivalue and 3) <= 1: # 00 or 01: no adjustment
      discard
    elif (ivalue and 4) == 0: # 10, even: no adjustment
      discard
    else: # 10, odd: no adjustment unless sticky is zero
      var sticky = lower
      if sticky == 0:
        while j > 0:
          dec j
          if acc.chunk[j] != 0:
            sticky = 1
            break
      if sticky == 0: # exact half → round to even (away from zero)
        ivalue += 4
        if (ivalue and (int64(1) shl (MantBits + 3))) != 0:
          ivalue = ivalue shr 1
          inc eFinal

  # Drop the two rounding bits and form the IEEE exponent field.
  ivalue = ivalue shr 2
  eFinal += (i shl LowExpBits) - ExpBias - MantBits

  if eFinal >= ExpMask: # overflow → ±Inf
    intv = intv or (uint64(ExpMask) shl MantBits)
    return fromBits[T](intv)

  intv = intv or (uint64(eFinal) shl MantBits)
  intv = intv or (uint64(ivalue) and MantMask) # mask out the implicit 1
  return fromBits[T](intv)

# ----------------------------------------------------------------------------
# Public convenience: the exact, correctly-rounded sum / dot of an array
# ----------------------------------------------------------------------------

func superSum*[T: SomeFloat](x: openArray[T]): T {.contractual.} =
  ## Exact, correctly-rounded sum of `x` via Neal's small superaccumulator: the
  ## real sum accumulated exactly and rounded once to nearest (ties-to-even).
  ## For finite, non-overflowing float64 input this equals `math.fsum`
  ## bit-for-bit and agrees with `nearSum`. NaN/±Inf propagate (IEEE); finite
  ## inputs never raise and never yield NaN — the exact integer accumulator has
  ## no `Inf − Inf` order artifact, so overflow gives correctly-signed ±Inf and
  ## cancellation gives the true finite value (the property `naiveDot`/`dot2`
  ## rely on for their finite-input NaN fallback). Online and fixed-memory: see
  ## `SuperAccumulator` for streaming use.
  ensure:
    x.len != 0 or result == T(0)
    not allFin(x) or classify(result) != fcNan # finite input ⇒ no NaN (overflow ⇒ ±Inf)
  body:
    if x.len == 0:
      return T(0)
    var acc: SuperAccumulator[T]
    initSuperAccumulator(acc)
    acc.add(x)
    result = acc.round()

func superDot*[T: SomeFloat](x, y: openArray[T]): T {.contractual.} =
  ## Exact, correctly-rounded dot product `Σ xᵢ·yᵢ` via the superaccumulator:
  ## each product is accumulated exactly with `addProduct` (no intermediate `T`
  ## rounding, so products that overflow `T` are held at their true magnitude),
  ## then rounded once to nearest (ties-to-even). Finite inputs ⇒ the result is
  ## the correctly-rounded dot — finite, or correctly-signed ±Inf on genuine
  ## overflow — and is *never* NaN, unlike `naiveDot`/`dot2`/`dotK`, whose
  ## opposite-sign ±Inf products can combine to `+Inf + −Inf = NaN`; it is their
  ## finite-input NaN fallback. NaN/±Inf operands propagate per IEEE. Online and
  ## fixed-memory.
  require:
    x.len == y.len
  ensure:
    x.len != 0 or result == T(0)
    not (allFin(x) and allFin(y)) or classify(result) != fcNan # finite input ⇒ no NaN
  body:
    if x.len == 0:
      return T(0)
    var acc: SuperAccumulator[T]
    initSuperAccumulator(acc)
    for i in 0 ..< x.len:
      acc.addProduct(x[i], y[i])
    result = acc.round()
