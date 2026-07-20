## C ABI for UniAccurate. Built --app:staticlib/--app:lib --noMain --mm:arc -d:release.
## Keep in sync with include/UniAccurate.h; tests/c links the header against this lib.
import ../UniAccurate

const UniAccurateVersionC: cstring = "0.0.1"

# Unmangled C symbols, C calling convention, exported from the shared lib.
{.push exportc, cdecl, dynlib.}

proc ua_fibonacci(n: cint): clonglong =
  ## fibonacci(n), n clamped to [0, FibMaxN]. n < 0 -> 0, n > FibMaxN -> fibonacci(FibMaxN). Never raises.
  let m = int(n)
  if m < 0:
    return clonglong(0)
  if m > FibMaxN:
    return fibonacci(FibMaxN).clonglong
  fibonacci(m).clonglong

proc ua_version(): cstring {.exportc, cdecl, dynlib.} =
  ## Static version string; do not free.
  UniAccurateVersionC

{.pop.}
