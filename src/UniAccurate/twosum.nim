## Error-free sum (Møller-Knuth TwoSum) — the foundation of the UniAccurate
## EFT layer. Replaces the template's fibonacci placeholder.
##
## `a + b = s + e` exactly in real arithmetic, with `s = fl(a + b)` (IEEE-754
## roundTiesToEven) and `|e| <= 1/2 ulp(s)` for normal `s`. 6 FLOPs, branchless,
## no precondition on operand ordering. Møller (1965); Knuth (1998) TAOCP 2,
## §4.2.2, Theorem C.

func twoSum*[T: SomeFloat](a, b: T): (T, T) {.inline.} =
  ## Møller-Knuth error-free addition. Returns `(s, e)` with `a + b = s + e`
  ## exactly and `s = fl(a + b)`.
  result[0] = a + b
  let z = result[0] - a
  result[1] = (a - (result[0] - z)) + (b - z)
