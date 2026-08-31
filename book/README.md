<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 lituus-lab -->
# The Book

A nimibook table of contents, one chapter per file. Every code block is
compiled and run when the book is built, so prose that outlives its API breaks
the build rather than quietly misleading a reader.

| File | What it is |
|---|---|
| `nbook.nim` | the table of contents and the theme selection — the driver |
| `nimib.toml` | nimib's own configuration, read from this directory |
| `config.nims` | the paths each chapter's own compilation needs |
| `index.nim` | what an error-free transformation is, and the Nim surface |
| `contracts.nim` | why the contract is part of the signature |
| `summation.nim` | naive, pairwise and compensated summation |
| `exact.nim` | correctly rounded, exact, and the faithful middle ground |
| `reductions.nim` | mean, variance and the rest, done accurately |
| `dot.nim` | the dot product, and what compensation buys |
| `surfaces.nim` | the C ABI and the Python binding |
| `references.nim` | the papers the algorithms come from |

## Building it

```bash
build/unigate book     # the book alone
build/unigate docs     # book + generated API reference, into pages/
```

Through the gate, never `nimble book` directly: nimble exits 0 even when an
`exec` inside a task fails, so a green run that went through it proves nothing.

`book` runs nimibook's `init` before `build`. `init` is what creates
`__site/assets`, which is not tracked: without it every page ships referencing
a stylesheet and a script that are not there.

## Adding a chapter

Add the entry to `nbook.nim`'s table of contents, then `nimble bookInit`
scaffolds the missing source.

Each chapter calls `nbInit(theme = useNimibook)` itself and then `useLituus()`.
`nbInit` cannot be wrapped: it reads `instantiationInfo(-1)` to learn which
file it is documenting. A Markdown entry never runs any Nim, so it never gets
the theme — keep every chapter a `.nim`.

The version appears in `index.nim` as the Python surface's output, inside a
fenced block the book does not execute. `tests/test_version.nim` reads it, so
it cannot drift away from the manifest unnoticed.
