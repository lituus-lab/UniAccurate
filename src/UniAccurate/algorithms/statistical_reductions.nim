# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Numerically stable reductions used by statistical consumers.
import std/math
import contracts
import ./exactsum

func cLdexp(x: cdouble; exponent: cint): cdouble {.
    importc: "ldexp", header: "<math.h>".}
func cLdexpf(x: cfloat; exponent: cint): cfloat {.
    importc: "ldexpf", header: "<math.h>".}

func scalePowerOfTwo[T: SomeFloat](value: T; exponent: int): T {.inline.} =
  when T is float64:
    T(cLdexp(cdouble(value), cint(exponent)))
  else:
    T(cLdexpf(cfloat(value), cint(exponent)))

func normalizedCentered[T: SomeFloat](values: openArray[T]; center: T):
    tuple[scale: T; deviations: seq[T]] =
  for value in values:
    result.scale = max(result.scale, max(abs(value), abs(center)))
  if classify(result.scale) in {fcNan, fcInf}:
    return
  result.deviations = newSeq[T](values.len)
  if result.scale == T(0):
    return
  for index, value in values:
    let deviation = value - center
    result.deviations[index] =
      if classify(deviation) notin {fcInf, fcNegInf}: deviation / result.scale
      else: value / result.scale - center / result.scale

func scaledMean*[T: SomeFloat](values: openArray[T]): T {.contractual.} =
  ## Exact-total mean when the sum is representable, with a quotient-scaled
  ## fallback when finite inputs overflow that intermediate sum.
  require:
    values.len > 0
  body:
    if values.len == 0:
      raise newException(ValueError, "mean requires at least one value")
    let total = superSum(values)
    var finiteInput = true
    for value in values:
      if classify(value) in {fcNan, fcInf, fcNegInf}:
        finiteInput = false
        break
    if classify(total) notin {fcInf, fcNegInf} or not finiteInput:
      return total / T(values.len)
    var scaled: SuperAccumulator[T]
    initSuperAccumulator(scaled)
    let divisor = T(values.len)
    for value in values:
      scaled.add(value / divisor)
    scaled.round()

func scaledEuclideanNorm*[T: SomeFloat](values: openArray[T]): T =
  ## BLAS LASSQ-style norm without intermediate square overflow or underflow.
  var scale = T(0)
  var scaledSquares = T(1)
  for value in values:
    let magnitude = abs(value)
    if classify(magnitude) == fcNan:
      return T(NaN)
    if classify(magnitude) == fcInf:
      return T(Inf)
    if magnitude != T(0):
      if scale < magnitude:
        let ratio = scale / magnitude
        scaledSquares = T(1) + scaledSquares * ratio * ratio
        scale = magnitude
      else:
        let ratio = magnitude / scale
        scaledSquares += ratio * ratio
  if scale == T(0): T(0) else: scale * sqrt(scaledSquares)

func centeredSumSquares*[T: SomeFloat](values: openArray[T]; center: T): T =
  ## Correctly rounded sum of the represented deviations squared.
  var deviations = newSeq[T](values.len)
  for index, value in values:
    deviations[index] = value - center
  superDot(deviations, deviations)

func centeredCrossProduct*[T: SomeFloat](x, y: openArray[T]; centerX,
    centerY: T): T {.contractual.} =
  ## Correctly rounded sum of represented centered pair products.
  require:
    x.len == y.len
  body:
    var deviationsX = newSeq[T](x.len)
    var deviationsY = newSeq[T](y.len)
    for index in 0 ..< x.len:
      deviationsX[index] = x[index] - centerX
      deviationsY[index] = y[index] - centerY
    superDot(deviationsX, deviationsY)

func centeredCosineSimilarity*[T: SomeFloat](x, y: openArray[T]; centerX,
    centerY: T): T {.contractual.} =
  ## Cosine similarity of centered vectors without overflowing raw products.
  require:
    x.len == y.len
    x.len > 0
  body:
    if x.len != y.len or x.len == 0:
      raise newException(ValueError,
        "centered vectors must have equal non-zero lengths")
    let
      normalizedX = normalizedCentered(x, centerX)
      normalizedY = normalizedCentered(y, centerY)
    if classify(normalizedX.scale) in {fcNan, fcInf} or
        classify(normalizedY.scale) in {fcNan, fcInf}:
      return T(NaN)
    let
      normX = scaledEuclideanNorm(normalizedX.deviations)
      normY = scaledEuclideanNorm(normalizedY.deviations)
    if normX == T(0) or normY == T(0):
      return T(NaN)
    let similarity = (superDot(normalizedX.deviations,
      normalizedY.deviations) / normX) / normY
    max(T(-1), min(T(1), similarity))

func centeredProjectionCoefficient*[T: SomeFloat](x, y: openArray[T]; centerX,
    centerY: T): T {.contractual.} =
  ## Range-safe `sum(dx*dy) / sum(dx*dx)` for a centered projection.
  require:
    x.len == y.len
    x.len > 0
  body:
    if x.len != y.len or x.len == 0:
      raise newException(ValueError,
        "centered vectors must have equal non-zero lengths")
    let
      normalizedX = normalizedCentered(x, centerX)
      normalizedY = normalizedCentered(y, centerY)
    if classify(normalizedX.scale) in {fcNan, fcInf} or
        classify(normalizedY.scale) in {fcNan, fcInf}:
      return T(NaN)
    let
      numerator = superDot(normalizedX.deviations, normalizedY.deviations)
      denominator = superDot(normalizedX.deviations, normalizedX.deviations)
    if denominator == T(0):
      return T(NaN)
    if numerator == T(0):
      return T(0)
    let
      numeratorParts = frexp(numerator)
      denominatorParts = frexp(denominator)
      scaleYParts = frexp(normalizedY.scale)
      scaleXParts = frexp(normalizedX.scale)
      mantissa = (numeratorParts.frac * scaleYParts.frac) /
        (denominatorParts.frac * scaleXParts.frac)
      exponent = numeratorParts.exp + scaleYParts.exp - denominatorParts.exp -
        scaleXParts.exp
    scalePowerOfTwo(mantissa, exponent)
