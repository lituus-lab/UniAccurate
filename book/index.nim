# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniAccurate

nbInit(theme = useNimibook)
useLituus()
nb.title = "UniAccurate"

nbText: """
# UniAccurate

Error-free transformations (EFT) for floating-point arithmetic, exposed across
three surfaces: **Nim**, a **C ABI**, and a **Python** binding.

An EFT computes an operation — here, addition — and returns both the rounded
result `s = fl(a + b)` and the exact rounding error `e`, such that in real
arithmetic `a + b = s + e` exactly. The error is not lost; compensated
summation threads it back through a running sum. This page is a nimib book:
every Nim block below is compiled and run when the book is built, and the
output shown is what the code actually produced. A change that breaks the API
breaks the docs build, so the two cannot drift apart.

## The Nim surface

The umbrella module re-exports every public submodule.
"""

nbCode:
  import UniAccurate

  echo "version ", UniAccurateVersion
  let (s, e) = twoSum(1.0, 2.0)
  echo "twoSum(1.0, 2.0) = (", s, ", ", e, ")"
  let (s2, e2) = twoSum(1.0, 2e16)
  echo "twoSum(1.0, 2e16) = (", s2, ", ", e2, ")"

nbText: """
`2e16 + 1` rounds back to `2e16` in float64 — the `1` is below the ULP, so a
plain `+` drops it. `twoSum` recovers it as `e = 1.0`: the identity `a + b =
s + e` still holds to the last bit.
"""

nbSave
