## C ABI for UniAccurate. Built --app:staticlib/--app:lib --noMain --mm:arc -d:release.
## Keep in sync with include/UniAccurate.h; tests/c links the header against this lib.
import std/math
import ../UniAccurate

const UniAccurateVersionC: cstring = "0.1.0"

# Unmangled C symbols, C calling convention, exported from the shared lib.
{.push exportc, cdecl, dynlib.}

proc ua_two_sum(a, b: cdouble; s, e: ptr cdouble) {.raises: [].} =
  ## Error-free sum: writes `*s = fl(a+b)` and `*e` with `a + b = *s + *e`
  ## exactly in real arithmetic (Møller–Knuth TwoSum). For non-finite `a` or
  ## `b`, `*s = a + b` (IEEE) and `*e = NaN`. Never raises. Null `s` or `e` is
  ## undefined.
  if classify(a) in {fcNan, fcInf, fcNegInf} or
      classify(b) in {fcNan, fcInf, fcNegInf}:
    s[] = a + b
    e[] = NaN
  else:
    let (ss, ee) = twoSum(a, b)
    s[] = ss
    e[] = ee

proc ua_version(): cstring {.exportc, cdecl, dynlib.} =
  ## Static version string; do not free.
  UniAccurateVersionC

{.pop.}
