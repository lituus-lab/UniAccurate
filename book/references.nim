# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
import nimib, nimibook
import lituus_theme
import UniAccurate

nbInit(theme = useNimibook)
useLituus()
nb.title = "References"

nbText: """
## References

The EFT identities, their exactness conditions, and the FMA contraction
invariant are documented in `src/UniAccurate/twosum.nim` with full citations
(Dekker 1971; Møller 1965; Knuth 1998; Ogita, Rump & Oishi 2005; Boldo &
Melquiond 2008; Shewchuk 1997; Goldberg 1991). The generated API reference
lists the symbols; this book is where the layer gets explained.
"""

nbSave

nbSave
