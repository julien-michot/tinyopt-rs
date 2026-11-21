// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Timing loop mirroring Catch2's benchmark methodology.

use std::time::{Duration, Instant};

/// Result of one solve.
#[derive(Clone, Copy, Debug)]
pub struct Outcome {
    pub converged: bool,
    pub iterations: u32,
    pub final_cost: f64,
}

/// Timing statistics.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub mean_ns: f64,
    pub median_ns: f64,
    pub min_ns: f64,
    pub samples: usize,
}

/// Run a benchmark with warmup and samples.
pub fn measure(
    warmup: Duration,
    samples: usize,
    mut run: impl FnMut() -> Outcome,
) -> (Stats, Outcome) {
    let mut last = run();
    let start = Instant::now();
    while start.elapsed() < warmup {
        last = run();
    }

    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t0 = Instant::now();
        last = run();
        times.push(t0.elapsed().as_nanos() as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = if times.len() % 2 == 0 {
        0.5 * (times[times.len() / 2 - 1] + times[times.len() / 2])
    } else {
        times[times.len() / 2]
    };
    (
        Stats {
            mean_ns: mean,
            median_ns: median,
            min_ns: times[0],
            samples: times.len(),
        },
        last,
    )
}
