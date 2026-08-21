# Statistical-reduction baseline

`nimble benchStatisticalReductions` compiles with release ORC and records three
runs of twenty complete calls after three warmups. Inputs are prepared outside
the windows. The norm window is allocation-free; centered windows deliberately
include deviation-buffer allocation and the exact superaccumulator.

Set `UNIACCURATE_BENCHMARK_MACHINE` when producing a versioned machine baseline.
The numbers detect local regressions and are not portable speedup claims.

On the recorded Apple M4, 100,000 values take median means of 0.4356 ms for
scaled norm, 0.5563 ms for centered squares, and 0.5798 ms for centered cross
product.
