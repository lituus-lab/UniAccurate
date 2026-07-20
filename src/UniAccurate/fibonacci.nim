## Iterative Fibonacci. Template hello-world, exercised in Nim/C/Python.
import contracts

const FibMaxN* = 92
  ## Largest n with fibonacci(n) fitting in int64 (fib(92) fits, fib(93) overflows).

func fibonacci*(n: int): int {.contractual.} =
  ## fibonacci(n) for n in [0, FibMaxN]. O(n) time, O(1) space.
  ##
  ## .. code-block:: nim
  ##   doAssert fibonacci(10) == 55
  require:
    n >= 0 and n <= FibMaxN
  ensure:
    result >= 0
  body:
    if n == 0: return 0
    if n == 1: return 1
    var a = 0
    var b = 1
    for _ in 2 .. n:
      let t = a + b
      a = b
      b = t
    result = b

