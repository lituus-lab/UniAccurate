# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Numerically stable reductions used by statistical consumers.
import std/math
import contracts
import ./exactsum

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
    var scaleX, scaleY = T(0)
    for index in 0 ..< x.len:
      scaleX = max(scaleX, max(abs(x[index]), abs(centerX)))
      scaleY = max(scaleY, max(abs(y[index]), abs(centerY)))
    if classify(scaleX) == fcNan or classify(scaleY) == fcNan:
      return T(NaN)
    if classify(scaleX) == fcInf or classify(scaleY) == fcInf:
      return T(NaN)
    var normalizedX = newSeq[T](x.len)
    var normalizedY = newSeq[T](y.len)
    for index in 0 ..< x.len:
      if scaleX != T(0):
        let deviation = x[index] - centerX
        normalizedX[index] =
          if classify(deviation) notin {fcInf, fcNegInf}: deviation / scaleX
          else: x[index] / scaleX - centerX / scaleX
      if scaleY != T(0):
        let deviation = y[index] - centerY
        normalizedY[index] =
          if classify(deviation) notin {fcInf, fcNegInf}: deviation / scaleY
          else: y[index] / scaleY - centerY / scaleY
    let
      normX = scaledEuclideanNorm(normalizedX)
      normY = scaledEuclideanNorm(normalizedY)
    if normX == T(0) or normY == T(0):
      return T(NaN)
    let similarity = (superDot(normalizedX, normalizedY) / normX) / normY
    max(T(-1), min(T(1), similarity))
