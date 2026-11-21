// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Result schema shared with `report.py`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub category: String,
    pub problem: String,
    pub backend: String,
    pub mean_ns: f64,
    pub median_ns: f64,
    pub min_ns: f64,
    pub samples: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationRecord {
    pub category: String,
    pub problem: String,
    pub backend: String,
    pub iterations: u32,
    pub converged: bool,
    pub final_cost: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Meta {
    pub generated_at: String,
    pub os: String,
    pub arch: String,
    pub rustc: String,
    pub threads: usize,
    pub samples: usize,
    pub warmup_ms: u64,
    pub backends: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Results {
    pub schema_version: u32,
    pub meta: Meta,
    pub records: Vec<Record>,
    pub iterations: Vec<IterationRecord>,
    pub notes: Vec<String>,
}
