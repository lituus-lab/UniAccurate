# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## The version, stated in seven places, checked to agree.
##
## Nimble refuses anything but a string literal for `version`, so the manifest
## cannot import a shared constant and no arrangement makes one file the source
## the others derive from. What is achievable is proof: this test reads every
## copy and fails when one drifts, which is what a release needs before it can
## claim manifest = header = wheel = tag.
import std/[unittest, os, strutils]
import UniAccurate

const Root = currentSourcePath().parentDir.parentDir

proc valueOf(path, key, opener, closer: string): string =
  ## The first `key … opener VALUE closer` on one line of the file; an empty
  ## `closer` reads to the end of the line. Deliberately crude: a parser per
  ## format would be more code than the thing it checks.
  for line in readFile(Root / path).splitLines:
    let at = line.find(key)
    if at < 0: continue
    let opens = line.find(opener, at + key.len)
    if opens < 0: continue
    let value = line[opens + opener.len .. ^1]
    if closer.len == 0: return value.strip
    let closes = value.find(closer)
    if closes < 0: continue
    return value[0 ..< closes]
  ""

suite "one version, seven copies":
  let manifest = valueOf("UniAccurate.nimble", "version", "\"", "\"")

  test "the manifest states one":
    check manifest.len > 0
    check manifest.count('.') == 2

  test "the Nim constant agrees":
    check UniAccurateVersion == manifest

  test "the C header agrees, macros and string alike":
    let parts = manifest.split('.')
    check valueOf("include/UniAccurate.h", "UNIACCURATE_VERSION_MAJOR", " ",
        "") == parts[0]
    check valueOf("include/UniAccurate.h", "UNIACCURATE_VERSION_MINOR", " ",
        "") == parts[1]
    check valueOf("include/UniAccurate.h", "UNIACCURATE_VERSION_PATCH", " ",
        "") == parts[2]
    check valueOf("include/UniAccurate.h", "define UNIACCURATE_VERSION ", "\"",
        "\"") == manifest

  test "the C ABI reports it":
    # Read from the source rather than called: this suite links no C library.
    check valueOf("src/UniAccurate/c_api.nim", "UniAccurateVersionC", "\"",
        "\"") == manifest

  test "the Python distribution agrees":
    check valueOf("py/pyproject.toml", "version", "\"", "\"") == manifest

  test "the book states it":
    # A version written into prose is a version that drifts. This one is the
    # Python surface's output, quoted in a fenced block the book does not
    # execute, so nothing else would notice.
    check valueOf("book/surfaces.nim", "uniaccurate.version()", "'", "'") == manifest

  test "the Python test expects it":
    check valueOf("py/tests/test_two_sum.py", "uniaccurate.version()", "\"",
        "\"") == manifest
