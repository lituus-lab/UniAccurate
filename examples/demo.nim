# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import std/strutils
import UniAccurate

echo "UniAccurate " & UniAccurateVersion
for (a, b) in [(1.0, 2.0), (1.0, 2e16), (1e20, 1.0)]:
  let (s, e) = twoSum(a, b)
  echo "twoSum(" & $a & ", " & $b & ") = (" & $s & ", " & $e & ")"
