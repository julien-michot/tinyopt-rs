# Performance

This page records how the library is benchmarked and the numbers we track. Treat
the absolute values as indicative: they were collected on an Apple Silicon laptop,
and the comparison harness runs each backend in its own process because thermal
drift is significant on long runs.

## How to reproduce

```sh
# Rust-native criterion suites (dense, sparse, block-sparse, block-tridiagonal)
pixi run bench

# Cross-library comparison (tinyopt-rs, Ceres, C++ tinyopt)
pixi run bench-compare
pixi run bench-report

# Same, with the optional CHOLMOD sparse backend
pixi run -e suitesparse cargo run --release --manifest-path benchmarks/compare/Cargo.toml \
    --features suitesparse -- run --backends tinyopt
# and with TINYOPT_BENCH_SOLVER=cholmod in the environment
```

The comparison harness writes `benchmarks/compare/results/results.json`; the
report script renders Markdown, HTML, CSV and plots into `results/report/`.

For an A/B of two commits, build the benchmark binary at each revision and run
them interleaved (base, new, base, new, ...), then compare medians. Running them
back-to-back does not cancel thermal drift.

## What we optimize for

- Dense static problems: no heap allocation in the optimization loop.
- Sparse scalar problems: constant CSC pattern across iterations, so the solving
  cost is factorization + solves, not assembly.
- Block-structured problems (bundle adjustment): the accumulation and the block
  Schur complement dominate; both are kept allocation-free and vectorized for
  the common fixed block shapes.

## Bundle adjustment (10 cameras, 3000 points)

| Variant | tinyopt-rs | C++ tinyopt | Ceres |
|---|---:|---:|---:|
| Block-sparse Hessian | ~37 ms | ~17 ms | ~130 ms |
| Scalar sparse Hessian | ~232 ms | ~192 ms | ~102 ms |

The block-sparse path is the one that improved the most in the last round
(78 → 37 ms, 2.1x):

- block storage is a contiguous `Vec` per type pair with O(1) index grids,
  instead of a `BTreeMap<(row, col), Vec<S>>` that allocated per block;
- the block pattern is analyzed once and only values are refreshed per iteration;
- the Schur complement uses compile-time sized tiles (6x6x3, 3x3x3, ...) and
  reuses its scratch buffers;
- `block_mut::<R, C>` returns a const-generic view, so `add_jtj` and
  `add_outer_product` unroll and vectorize.

## Sparse chains

| Size | tinyopt-rs | C++ tinyopt | Ceres |
|---|---:|---:|---:|
| 100D | 0.04 ms | 0.04 ms | 0.37 ms |
| 1000D | 0.87 ms | 1.95 ms | 3.79 ms |

CHOLMOD (optional `suitesparse` feature) is 1.04-1.11x faster on chains and
within noise on bundle adjustment, where accumulation dominates. Solver-level
benchmarks (`cargo bench --bench sparse --features suitesparse`) show 1.7-2.8x
faster refactorize + solve, and up to 11x faster cold solves on large 2-D grids.

## Solver choice (sparse chain)

| Size | Levenberg-Marquardt | Steihaug–Toint (Newton-CG) |
|---|---:|---:|
| 100D | 83 µs | 65 µs |
| 1000D | 2.32 ms | 1.35 ms |

Newton-CG only needs Hessian-vector products, so it avoids forming and
factorizing the normal equations; it wins on the chain problems and scales
better with dimension. `LinearSolverMethod::IncompleteCholesky` provides an
IC(0)-preconditioned CG solver; it is an *iterative* option, so the direct
sparse LDLᵀ is still faster on small and medium systems and IC-PCG is meant for
very large problems where the factorization dominates memory.

## Block-tridiagonal chains (fixed 6x6 blocks)

| Blocks | tinyopt-rs | Ceres (sparse normal equations) |
|---|---:|---:|
| 32 | 77 µs | 246 µs |
| 64 | 153 µs | 508 µs |
| 128 | 536 µs | 1.08 ms |

Both converge to the same final cost; the dedicated block-tridiagonal solver
exploits the banded structure instead of factorizing a general sparse matrix.

## Interpreting the numbers

- Times are means of the harness samples; criterion reports use medians.
- The C++ binary uses its own default options; the Rust harness mirrors
  `benchmarks/common/options.h` so the stopping criteria match.
- Ceres fixes the gauge (camera 0, point 0) as in the C++ Ceres benchmark.
- Runs on a loaded machine can be 2x slower in absolute terms; compare ratios,
  not raw values, across sessions.
