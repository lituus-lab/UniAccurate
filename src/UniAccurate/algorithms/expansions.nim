# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Shewchuk expansion arithmetic.
##
## An *expansion* is a sequence of non-overlapping floats, ascending in
## magnitude, that together represent a real value exactly: each component
## occupies a disjoint set of bit positions, so the real sum of the components
## is the represented value with no rounding. This module implements the
## expansion-level primitives of Shewchuk (1997): growing an expansion by one
## component, adding two expansions, scaling an expansion by a scalar, and
## reading off the nearest float. The EFT primitives (`twoSum`, `fastTwoSum`,
## `twoProduct`, `split`) they are built on live in `twosum`.
##
## Representation invariant. A *valid* expansion `e` of length `n` has, for all
## `i < j`, `isNonOverlapping(e[i], e[j])` and `abs(e[i]) <= abs(e[j])`. The
## ZeroElim variants additionally drop zero components. The algorithms here
## preserve the invariant; the tests check it.
##
## Correctness assumptions. Inherit `twosum`'s: roundTiesToEven,
## `FLT_EVAL_METHOD == 0`, no `-ffp-contract`/`-ffast-math`, and finite inputs
## whose intermediates neither overflow nor (for `twoProduct`) underflow to a
## non-representable product. Under those the expansion is the exact real
## result; `estimate` then reads it to within one ulp, `shewchukSum` to the
## last bit.
##
## References
## ==========
## - Shewchuk, J.R. (1997). "Adaptive Precision Floating-Point Arithmetic and
##   Fast Robust Geometric Predicates". *Discrete Comput. Geom.* 18(3),
##   305–363. doi:10.1007/PL00009321
## - Dekker, T.J. (1971). "A floating-point technique for extending the
##   available precision". *Numer. Math.* 18(3), 224–242.
##   doi:10.1007/BF01397083
import std/math
import contracts
import ../twosum

const
  splitter* = SplitFactor64
    ## Shewchuk's name for the Veltkamp split factor (`SplitFactor64`, `2^27+1`).
    ## Splits a 53-bit float64 significand into 26+27 non-overlapping bits.
  machineEpsilon* = 2.220446049250313e-16
    ## Float64 machine epsilon `2^-52` = `2 * ulp(1.0)`, twice the unit roundoff.

func twoSquare*[T: SomeFloat](a: T): (T, T) {.contractual, inline.} =
  ## Error-free square: `a * a = p + e` exactly, `p = fl(a*a)`, `|e| < ulp(p)`.
  ## One Veltkamp split (vs `twoProduct`'s two): the cross terms `ah*al + al*ah`
  ## coincide, so their exact sum `2*(ah*al)` is formed with a single product.
  ## Dekker (1971); Shewchuk (1997) Two_Square.
  require:
    classify(a) in {fcNormal, fcSubnormal, fcZero, fcNegZero}
    a == T(0) or a * a != T(0) # reject total product underflow
  ensure:
    classify(result[0]) in {fcInf, fcNegInf, fcNan} or
      classify(result[1]) in {fcNan} or
      abs(result[1]) <= ulp(result[0])
  body:
    result[0] = a * a
    let (ah, al) = split(a)
    result[1] = ((ah * ah - result[0]) + 2 * (ah * al)) + al * al

func twoDiffTail*[T: SomeFloat](a, b, s: T): T {.contractual, inline.} =
  ## Tail of `twoDiff` given the precomputed head `s = fl(a - b)`: returns `e`
  ## with `a - b = s + e` exactly, `|e| <= ulp(s)`. The head/tail split lets a
  ## caller that already has `s` recover the error without recompute. Knuth
  ## (1998) TAOCP 2, §4.2.2.
  ensure:
    # Mirror twoDiff's postcondition (head = s): the bound holds for finite s,
    # and the non-finite short-circuit avoids reading ulp(±Inf) = NaN.
    classify(s) in {fcInf, fcNegInf, fcNan} or abs(result) <= ulp(s)
  body:
    let z = s - a
    (a - (s - z)) - (b + z)

func zeroElim*[T: SomeFloat](e: openArray[T]): seq[T] =
  ## Drop the zero components of `e`, preserving order. A zero-eliminated valid
  ## expansion stays valid (zeros carry no bits, so removing them cannot create
  ## overlap or reorder magnitudes).
  result = @[]
  for v in e:
    if v != T(0):
      result.add(v)

func growExpansion*[T: SomeFloat](e: openArray[T], b: T): seq[T] =
  ## Grow `e` by one component `b`: returns an expansion `h` (length `e.len+1`,
  ## ascending, non-overlapping) whose real sum is `b + sum(e)` exactly. Zeros
  ## are NOT eliminated (feed the result to a ZeroElim primitive to compress).
  ## Shewchuk (1997) grow_expansion.
  result = newSeq[T](e.len + 1)
  var q = b
  for k in 0 ..< e.len:
    let (s, tail) = twoSum(q, e[k])
    result[k] = tail
    q = s
  result[e.len] = q

func fastExpansionSumZeroElim*[T: SomeFloat](e, f: openArray[T]): seq[T] =
  ## Add two expansions `e`, `f`: returns an expansion (ascending, non-
  ## overlapping, zero-eliminated) whose real sum is `sum(e) + sum(f)` exactly.
  ## Merges smallest-first: Shewchuk's sign-magnitude idiom
  ## `(f[fnow] > e[enow]) == (f[fnow] > -e[enow])` picks the smaller head of the
  ## two streams without `abs`. The first pair uses `fastTwoSum` (the second-
  ## smallest dominates the smallest, so `|a| >= |b|` holds); the rest use
  ## `twoSum` (no precondition). Tails are appended as the merge proceeds small
  ## to large, then the accumulated head last, so the result is ascending. A
  ## lone `[0]` is returned when the sum is exactly zero. Shewchuk (1997)
  ## fast_expansion_sum_zeroelim.
  if e.len == 0:
    return zeroElim(f)
  if f.len == 0:
    return zeroElim(e)
  var enow = 0
  var fnow = 0
  var h: seq[T] = @[]
  var q: T

  template takeSmallest(): T =
    if enow < e.len and (fnow >= f.len or
        ((f[fnow] > e[enow]) == (f[fnow] > -e[enow]))):
      let v = e[enow]
      inc enow
      v
    else:
      let v = f[fnow]
      inc fnow
      v

  q = takeSmallest()
  if enow < e.len or fnow < f.len:
    var (s, err) = fastTwoSum(takeSmallest(), q)
    q = s
    if err != T(0):
      h.add(err)
    while enow < e.len or fnow < f.len:
      (s, err) = twoSum(q, takeSmallest())
      q = s
      if err != T(0):
        h.add(err)
  if q != T(0) or h.len == 0:
    h.add(q)
  result = h

func scaleExpansionZeroElim*[T: SomeFloat](e: openArray[T], b: T): seq[T] =
  ## Scale `e` by `b`: returns an expansion (ascending, non-overlapping,
  ## zero-eliminated) whose real sum is `b * sum(e)` exactly. Processes `e`
  ## smallest-first; `fastTwoSum(p, s1)` is valid because `p = fl(e[i]*b)`
  ## dominates the accumulated head `s1` of the smaller products (Shewchuk's
  ## invariant for ascending non-overlapping `e`). A lone `[0]` is returned when
  ## the product is exactly zero. Shewchuk (1997) scale_expansion_zeroelim.
  if e.len == 0:
    return @[]
  var h: seq[T] = @[]
  var (q, qTail) = twoProduct(e[0], b)
  if qTail != T(0):
    h.add(qTail)
  for i in 1 ..< e.len:
    let (p, pTail) = twoProduct(e[i], b)
    let (s1, t1) = twoSum(q, pTail)
    if t1 != T(0):
      h.add(t1)
    let (s2, t2) = fastTwoSum(p, s1)
    if t2 != T(0):
      h.add(t2)
    q = s2
  if q != T(0) or h.len == 0:
    h.add(q)
  result = h

func estimate*[T: SomeFloat](e: openArray[T]): T =
  ## Nearest-float approximation of the expansion's real value, within one ulp:
  ## a smallest-first naive sum. Not correctly rounded in general — a tie or
  ## absorption can leave it one ulp from `shewchukSum`; use `shewchukSum` for
  ## the last bit. Shewchuk (1997) estimate.
  if e.len == 0:
    return T(0)
  result = e[0]
  for k in 1 ..< e.len:
    result = result + e[k]
