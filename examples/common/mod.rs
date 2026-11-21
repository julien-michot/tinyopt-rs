// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for the examples.

use tinyopt_rs::Real;

/// Format a scalar slice as comma-separated values.
pub fn fmt_slice<S: Real>(v: &[S]) -> String {
    v.iter()
        .map(|x| format!("{:.6}", x.to_f64().unwrap_or(f64::NAN)))
        .collect::<Vec<_>>()
        .join(", ")
}
