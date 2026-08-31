# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniAccurate

nbInit(theme = useNimibook)
useLituus()
nb.title = "The contract is part of the signature"

nbText: """
## The contract is part of the signature

`twoSum` is the Møller–Knuth form: six FLOPs, branchless, no precondition on
operand ordering. Its postcondition states the non-overlap bound `|e| <=
½ ulp(s)` for normal `s` — a property cheaper to check than the body is to
run, and never re-derived by calling the function again.

The contract is written with NimContracts (`require:` / `ensure:` / `body:`).
Under `-d:release` it compiles away entirely: the release build pays nothing,
while debug builds and the test suite catch a violation at the call site.
"""

nbText: """
"""

nbSave
