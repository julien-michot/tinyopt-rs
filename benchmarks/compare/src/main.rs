// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Cross-library benchmark harness: tinyopt-rs, Ceres (Rust binding) and the
//! C++ tinyopt binary on identical problems.

mod backends;
mod harness;
mod options;
mod problems;
mod results;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use results::{IterationRecord, Meta, Record, Results};

const TINYOPT: &str = "tinyopt-rs";
const CERES: &str = "ceres-rs";
const CPP: &str = "cpp-tinyopt";

#[derive(Clone, Debug)]
struct Args {
    out: PathBuf,
    samples: usize,
    warmup_ms: u64,
    quick: bool,
    full: bool,
    backends: Vec<String>,
    cpp_bench: PathBuf,
}

impl Default for Args {
    fn default() -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            out: manifest.join("results"),
            samples: 10,
            warmup_ms: 100,
            quick: false,
            full: false,
            backends: vec!["tinyopt".into(), "ceres".into(), "cpp".into()],
            cpp_bench: manifest
                .join("../../../exp-mop/build-bench/benchmarks/tinyopt/tinyopt_bench_tinyopt"),
        }
    }
}

fn main() {
    let mut argv = std::env::args().skip(1);
    let command = argv.next().unwrap_or_else(|| "run".to_string());
    let args = parse_args(argv.collect());
    match command.as_str() {
        "run" => run(args),
        "list" => list(&args),
        "check" => check(&args),
        other => {
            eprintln!("unknown command: {other}");
            eprintln!("usage: tinyopt-bench-compare [run|list|check] [OPTIONS]");
            std::process::exit(2);
        }
    }
}

fn parse_args(argv: Vec<String>) -> Args {
    let mut args = Args::default();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--out" => {
                i += 1;
                args.out = PathBuf::from(argv.get(i).expect("--out needs a value"));
            }
            "--samples" => {
                i += 1;
                args.samples = argv
                    .get(i)
                    .expect("--samples needs a value")
                    .parse()
                    .expect("valid sample count");
            }
            "--warmup-ms" => {
                i += 1;
                args.warmup_ms = argv
                    .get(i)
                    .expect("--warmup-ms needs a value")
                    .parse()
                    .expect("valid warmup");
            }
            "--quick" => args.quick = true,
            "--full" => args.full = true,
            "--backends" => {
                i += 1;
                let value = argv.get(i).expect("--backends needs a value");
                args.backends = value.split(',').map(str::to_string).collect();
            }
            "--cpp-bench" => {
                i += 1;
                args.cpp_bench = PathBuf::from(argv.get(i).expect("--cpp-bench needs a value"));
            }
            other => {
                eprintln!("unknown option: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    args
}

fn list(args: &Args) {
    println!("{:<26} {:<18} CATEGORY", "CASE", "SHORT");
    for case in problems::cases(args.full, args.quick) {
        println!(
            "{:<26} {:<18} {}",
            case.name,
            case.short,
            case.kind.category()
        );
    }
}

fn check(args: &Args) {
    for case in problems::cases(args.full, args.quick) {
        match case.kind {
            problems::Kind::BundleAdjustmentBlock { cams, points }
            | problems::Kind::BundleAdjustmentSparse { cams, points } => {
                let problem = problems::bundle_adjustment::make_problem(cams, points);
                println!(
                    "{:<26} {} observations, {} parameters, initial cost {:.12}",
                    case.name,
                    problem.observations.len(),
                    6 * problem.camera_count() + 3 * problem.point_count(),
                    problems::bundle_adjustment::initial_cost(&problem),
                );
            }
            problems::Kind::RobustPose { obs, robust } => {
                let truth = problems::robust_pose::ground_truth();
                let outliers = obs / 4;
                let data = problems::robust_pose::make_observations(
                    &truth,
                    obs - outliers,
                    outliers,
                    0.01,
                    50.0,
                );
                let mut cost = 0.0;
                for observation in &data {
                    let residual = truth.transform(&observation.world_point);
                    let dx = residual[0] - observation.measurement[0];
                    let dy = residual[1] - observation.measurement[1];
                    let dz = residual[2] - observation.measurement[2];
                    let squared = dx * dx + dy * dy + dz * dz;
                    if robust && squared > 0.09 {
                        cost += 0.5 * (2.0 * 0.3 * squared.sqrt() - 0.09);
                    } else {
                        cost += 0.5 * squared;
                    }
                }
                println!(
                    "{:<26} {} observations at the ground truth ({} loss), cost {:.6}",
                    case.name,
                    data.len(),
                    if robust { "Huber" } else { "L2" },
                    cost
                );
            }
            _ => {}
        }
    }
}

fn run(args: Args) {
    let cases = problems::cases(args.full, args.quick);
    let warmup = Duration::from_millis(args.warmup_ms);
    let mut records: Vec<Record> = Vec::new();
    let mut iterations: Vec<IterationRecord> = Vec::new();
    let mut backend_info: BTreeMap<String, String> = BTreeMap::new();

    for backend in &args.backends {
        match backend.as_str() {
            "tinyopt" => {
                backend_info.insert(
                    TINYOPT.into(),
                    format!("tinyopt {} (nalgebra)", env!("CARGO_PKG_VERSION")),
                );
                for case in &cases {
                    let (stats, outcome) =
                        harness::measure(warmup, args.samples, || backends::tinyopt::run(case));
                    records.push(Record {
                        category: case.kind.category().into(),
                        problem: case.name.clone(),
                        backend: TINYOPT.into(),
                        mean_ns: stats.mean_ns,
                        median_ns: stats.median_ns,
                        min_ns: stats.min_ns,
                        samples: stats.samples,
                    });
                    iterations.push(IterationRecord {
                        category: case.kind.category().into(),
                        problem: case.short.clone(),
                        backend: TINYOPT.into(),
                        iterations: outcome.iterations,
                        converged: outcome.converged,
                        final_cost: Some(outcome.final_cost),
                    });
                    print_line(TINYOPT, case, stats.mean_ns);
                }
            }
            "ceres" => {
                backend_info.insert(CERES.into(), "ceres-solver 0.5 (vendored Ceres 2.2)".into());
                for case in &cases {
                    let (stats, outcome) =
                        harness::measure(warmup, args.samples, || backends::ceres::run(case));
                    records.push(Record {
                        category: case.kind.category().into(),
                        problem: case.name.clone(),
                        backend: CERES.into(),
                        mean_ns: stats.mean_ns,
                        median_ns: stats.median_ns,
                        min_ns: stats.min_ns,
                        samples: stats.samples,
                    });
                    iterations.push(IterationRecord {
                        category: case.kind.category().into(),
                        problem: case.short.clone(),
                        backend: CERES.into(),
                        iterations: outcome.iterations,
                        converged: outcome.converged,
                        final_cost: Some(outcome.final_cost),
                    });
                    print_line(CERES, case, stats.mean_ns);
                }
            }
            "cpp" => {
                let warmup_seconds = args.warmup_ms as f64 / 1000.0;
                match backends::cpp::run(&args.cpp_bench, args.samples, warmup_seconds) {
                    Some(results) => {
                        backend_info.insert(CPP.into(), args.cpp_bench.display().to_string());
                        let mut missing = 0;
                        for case in &cases {
                            match results.means.get(&case.name) {
                                Some(&mean) => {
                                    records.push(Record {
                                        category: case.kind.category().into(),
                                        problem: case.name.clone(),
                                        backend: CPP.into(),
                                        mean_ns: mean,
                                        median_ns: mean,
                                        min_ns: mean,
                                        samples: args.samples,
                                    });
                                    print_line(CPP, case, mean);
                                }
                                None => missing += 1,
                            }
                            if let Some(&(count, converged)) = results
                                .iterations
                                .get(&(case.kind.category().to_string(), case.short.clone()))
                            {
                                iterations.push(IterationRecord {
                                    category: case.kind.category().into(),
                                    problem: case.short.clone(),
                                    backend: CPP.into(),
                                    iterations: count,
                                    converged,
                                    final_cost: None,
                                });
                            }
                        }
                        if missing > 0 {
                            eprintln!(
                                "warning: {missing} case(s) missing from the C++ binary output"
                            );
                        }
                    }
                    None => eprintln!(
                        "warning: C++ benchmark binary not found or failed: {}",
                        args.cpp_bench.display()
                    ),
                }
            }
            other => eprintln!("warning: unknown backend {other}"),
        }
    }

    let meta = Meta {
        generated_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_default(),
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        rustc: Command::new("rustc")
            .arg("--version")
            .output()
            .ok()
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".into()),
        threads: 1,
        samples: args.samples,
        warmup_ms: args.warmup_ms,
        backends: backend_info,
    };
    let results = Results {
        schema_version: 1,
        meta,
        records,
        iterations,
        notes: vec![
            "All Rust problems use the same data generators as the C++ suite (BA is noise-free and identical).".into(),
            "tinyopt-rs poses use an SE(3) retraction; ceres-rs uses a quaternion + translation with in-cost normalization (the safe binding has no manifold API).".into(),
            "Ceres fixes camera 0 and point 0 in bundle adjustment (gauge), as in the C++ Ceres benchmark.".into(),
            "ceres-rs has no ParameterBlockOrdering; SPARSE_SCHUR uses Ceres' automatic ordering.".into(),
            "Robust-pose noise uses a deterministic Rust PRNG (the C++ binary uses std::mt19937), so the data is statistically equivalent, not identical.".into(),
            "The C++ tinyopt 1D static benchmark times a scalar linear cost but reports iterations from its quadratic verification run; tinyopt-rs and ceres-rs optimize the quadratic residual for every dense case.".into(),
        ],
    };

    std::fs::create_dir_all(&args.out).expect("create output directory");
    let path = args.out.join("results.json");
    let json = serde_json::to_string_pretty(&results).expect("serialize results");
    std::fs::write(&path, json).expect("write results.json");
    println!(
        "\nSaved {} ({} records, {} iterations)",
        path.display(),
        results.records.len(),
        results.iterations.len()
    );
}

fn print_line(backend: &str, case: &problems::Case, mean_ns: f64) {
    let (value, unit) = if mean_ns >= 1e6 {
        (mean_ns / 1e6, "ms")
    } else {
        (mean_ns / 1e3, "µs")
    };
    println!("{backend:<12} {:<24} {:>10.2} {unit}", case.name, value);
}
