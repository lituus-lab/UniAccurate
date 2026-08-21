# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Numerically stable reductions used by statistical consumers.
import std/math
import contracts
import ./exactsum

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
