# tinyopt benchmark comparison

Compares **tinyopt-rs**, **Ceres** (Rust binding) and the **C++ tinyopt** binary on
the same problems, then renders plots plus an HTML/Markdown report.

The benchmark problems are direct ports of the C++ suite
(`benchmarks/tinyopt/*.cpp` and `benchmarks/common/*.h`): same data generators,
same stopping criteria, same solver settings.

## Quick start

```sh
# Build and run the short suite (one representative case per family)
pixi run bench-quick

# Full suite (all sizes except the 100c/50kp bundle adjustment, see --full)
pixi run bench-compare

# Render the report (HTML, plots, CSV) from the saved results
pixi run bench-report

# Markdown variant
pixi run -e bench python benchmarks/compare/report.py \
    --input benchmarks/compare/results/results.json --format md
```

Results and the report are written under `benchmarks/compare/results/`.

### Rayon bundle-adjustment comparison

The optional parallel benchmark compares 1 and 8 Rayon threads on the same
100-camera / 50,000-point block-sparse BA problem. It reports both direct
block-Schur compute + solve timings on a BA-shaped Hessian and end-to-end
optimization timings:

```sh
cargo run --release --manifest-path benchmarks/compare/Cargo.toml \
    --no-default-features --features rayon -- \
    parallel-ba --samples 5 --warmup-ms 100
```

The thread pool is created outside each timed sample. The end-to-end run uses
the parallel observation accumulator and the block-sparse Schur solver, so only
the configured pool size changes between the two measurements. Ceres is omitted
from this tinyopt-only benchmark build.

### Optional wgpu bundle adjustment

The benchmark package also supports GPU BA normal-equation assembly:

```sh
cargo run --release --manifest-path benchmarks/compare/Cargo.toml \
    --no-default-features --features wgpu -- \
    parallel-ba --samples 5 --warmup-ms 100 --wgpu-precision f32
```

Use `--wgpu-precision f64` to request native f64 assembly. This requires an
adapter exposing `wgpu::Features::SHADER_F64`; the shader uses quaternion-based
SE(3) projection and does not rely on f64 trigonometry. The f32 mode uses
portable WGSL for direct residual, Jacobian, and normal-equation assembly with
SE(3) camera states (unit quaternion + translation, right-tangent updates).
Both precision modes keep trial-cost evaluation, the block-sparse Schur solve,
and LM on the CPU.

## Backends

| Backend | How it runs |
|---|---|
| `tinyopt-rs` | In-process optimizer (`nalgebra` backend). |
| `ceres-rs` | In-process via the [`ceres-solver`](https://crates.io/crates/ceres-solver) crate, built from its vendored Ceres 2.2 + Eigen 3.4 sources (`source` feature). Needs `cmake` and a C++17 compiler; the Pixi benchmark environment builds the CPU-only Ceres library with a matching C/C++ toolchain. The first build takes a few minutes and is cached. |
| `cpp-tinyopt` | Runs the C++ `tinyopt_bench_tinyopt` binary and parses its Catch2 XML output. |

If Ceres was previously built with a different compiler setup, clear the old
benchmark cache once:

```sh
pixi run -e bench cargo clean --manifest-path benchmarks/compare/Cargo.toml
```

Build the C++ binary from the C++ checkout:

```sh
cd ../exp-mop && pixi run build-bench
```

The harness looks for `../exp-mop/build-bench/benchmarks/tinyopt/tinyopt_bench_tinyopt`;
override with `--cpp-bench <path>`. Missing binaries are skipped with a warning and
reported as `—`.

### CHOLMOD variant

Build with the `suitesparse` feature and set `TINYOPT_BENCH_SOLVER=cholmod` to route
tinyopt-rs sparse systems through CHOLMOD (the build needs SuiteSparse, e.g. the
`pixi run -e suitesparse` environment):

```sh
pixi run -e suitesparse cargo run --release --manifest-path benchmarks/compare/Cargo.toml \
    --features suitesparse -- run --backends tinyopt --filter sparse
```

## Benchmark matrix

| Category | Cases |
|---|---|
| Dense static | 1D/2D/3D double and float |
| Dense dynamic | 1D/2D/3D double and float; prior 6/12/33/50 |
| Sparse | chain 10/100/1000 |
| Bundle adjustment | block-sparse and scalar sparse, 2c/200p, 4c/600p, 10c/3kp (100c/50kp with `--full`) |
| Robust pose | 20/50/100 observations × {L2, Huber} |

## Solver settings

tinyopt-rs mirrors `benchmarks/common/options.h` and the C++ benchmark values:

- `max_iters = 100`, `min_error = 1e-12`, `min_rerr_dec = 1e-6`,
  `min_step_norm2 = 1e-16`, `min_grad_max = 1e-9`, `max_consec_failures = 10`
  (20 for the robust pose), `lm.jacobi_scaling = true`.
- Dense float cases use `min_error = 1e-6`, `min_step_norm2 = 1e-10`; the prior
  cases use `max_iters = 20`.

Ceres uses the same mapping as the C++ Ceres benchmarks: `DENSE_NORMAL_CHOLESKY`
(dense/pose), `SPARSE_NORMAL_CHOLESKY` (sparse chain), `SPARSE_SCHUR` (bundle
adjustment), LM trust region with `initial_trust_region_radius = 1e4`,
`function_tolerance = 1e-6`, `gradient_tolerance = 1e-9`,
`parameter_tolerance = 1e-8`, `min_relative_decrease = 1e-12`, one thread.

## Known differences

These are reported in every generated report:

- **Pose parameterization**: tinyopt-rs uses an SE(3) retraction (quaternion +
  translation, right update). The Rust Ceres binding has no manifold API, so
  `ceres-rs` optimizes normalized quaternion + translation blocks; its analytic
  Jacobian includes the normalization derivative, with the radial quaternion
  direction remaining null. The C++ tinyopt binary uses Sophus SE(3).
- **Gauge fixing**: the Ceres bundle-adjustment benchmark fixes camera 0 and
  point 0; the C++ tinyopt benchmark instead marks them fixed through its
  parameter traits. Both tinyopt-rs and the C++ binary fix the same parameters.
- **Ordering**: the Rust Ceres binding exposes no `ParameterBlockOrdering`, so
  `SPARSE_SCHUR` uses Ceres' automatic ordering.
- **Pose noise**: the Rust generator uses a deterministic local PRNG; the C++
  binary uses `std::mt19937`. Sizes, outlier ratios and noise levels match, the
  draws do not.
- **Block solver**: tinyopt-rs builds the bundle-adjustment Schur complement
  with dense per-block operations (independent point blocks); the C++ block
  engine additionally uses block AMD ordering and a general coupled-block
  elimination.
- **1D static**: the C++ tinyopt benchmark *times* a scalar linear cost but
  reports iterations from its quadratic verification run. tinyopt-rs and
  ceres-rs optimize the same quadratic residual for every dense case, so
  iteration counts are comparable.

## CLI

```
tinyopt-bench-compare run   [--out DIR] [--samples N] [--warmup-ms MS]
                            [--quick] [--full]
                            [--backends tinyopt,ceres,cpp] [--cpp-bench PATH]
tinyopt-bench-compare list  [--quick] [--full]
tinyopt-bench-compare check [--quick] [--full]   # problem sizes and initial costs
```

## Report

`report.py` reads `results.json` and writes:

- `benchmark-report.html` — styled report with embedded plots,
- `benchmark-report.md` — same content as Markdown (plot files referenced),
- `benchmark-results.csv`, `benchmark-iterations.csv`,
- `benchmark-{1..4}.png` — log-scale runtime plots per group.

```sh
python report.py --input results/results.json --output-dir results/report --format html
python report.py --input results/results.json --format md
```
