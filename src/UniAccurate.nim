# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## UniAccurate — umbrella module. Re-exports every public submodule.
import UniAccurate/twosum
import UniAccurate/algorithms/naivesum
import UniAccurate/algorithms/pairwisesum
import UniAccurate/algorithms/compensatedsum
import UniAccurate/algorithms/shewchuksum
import UniAccurate/algorithms/exactsum
import UniAccurate/algorithms/orosum
import UniAccurate/algorithms/dotproduct
import UniAccurate/algorithms/expansions
import UniAccurate/algorithms/statistical_reductions
export twosum
export naivesum
export pairwisesum
export compensatedsum
export shewchuksum
export exactsum
export orosum
export dotproduct
export expansions
export statistical_reductions
when defined(simd):
  import UniAccurate/simd
  export simd

const UniAccurateVersion* = "1.2.0"
