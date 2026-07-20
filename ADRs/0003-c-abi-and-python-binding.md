<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# ADR-0003: C ABI and Python binding

- Status: Accepted
- Date: 2026-07-15
- Scope: UniAccurate

## Decision

The Nim library is the source of truth; a hand-written C ABI
(`src/UniAccurate/c_api.nim`) and a hand-written header
(`include/UniAccurate.h`) expose it to non-Nim callers, built
`--app:staticlib`/`--app:lib --noMain --mm:arc -d:release`:

- The header is hand-written, not `--header:`-generated, and kept in sync by
  hand with `c_api.nim`. `tests/c` links the header against the built lib —
  a renamed or retyped symbol fails to link, so the C test doubles as the ABI
  drift detector between the two.
- `--mm:arc`: deterministic reference counting, no cycle collector to
  surprise a foreign caller holding a pointer across calls. `--noMain`: the
  library is linked into a host process that has its own `main`.
- The C ABI never raises a Nim exception across the boundary.

The Python binding (`py/`) is a Cython extension over the same C ABI, not a
`ctypes` wrapper: Cython gives static typing at the boundary and a compiled,
importable extension module. The shared library ships bundled inside the
wheel, found at import time via an RPATH relative to the extension
(`$ORIGIN` on Linux, `@loader_path` on macOS, `@rpath` install name) — no
separate `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` setup needed after
`pip install`.
