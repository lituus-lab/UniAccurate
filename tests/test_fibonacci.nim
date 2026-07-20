import std/unittest
import UniAccurate

suite "fibonacci":
  test "base cases":
    check fibonacci(0) == 0
    check fibonacci(1) == 1
    check fibonacci(2) == 1

  test "known values":
    check fibonacci(10) == 55
    check fibonacci(20) == 6765
    check fibonacci(50) == 12586269025
    check fibonacci(90) == 2880067194370816120

  test "max in-range (n = FibMaxN fits in int64)":
    let f = fibonacci(FibMaxN)
    check f == 7540113804746346429
    check f > 0

  test "recurrence fib(n) == fib(n-1) + fib(n-2) over [2, FibMaxN]":
    for n in 2 .. FibMaxN:
      check fibonacci(n) == fibonacci(n - 1) + fibonacci(n - 2)

  test "monotone non-decreasing":
    var prev = 0
    for n in 1 .. FibMaxN:
      let cur = fibonacci(n)
      check cur >= prev
      prev = cur
