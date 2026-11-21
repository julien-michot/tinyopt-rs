// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Optimization summary and per-iteration history, mirroring tinyopt's `Summary`.

use crate::cost::Cost;
use crate::stop_reasons::StopReason;

/// Per-iteration results.
#[derive(Clone, Debug, Default)]
pub struct History {
    /// Mean squared accumulated errors of all iterations.
    pub errs: Vec<f64>,
    /// Squared step norms of all iterations.
    pub deltas2: Vec<f64>,
    /// Step acceptance status.
    pub successes: Vec<bool>,
}

impl History {
    /// Record one iteration.
    pub fn add(&mut self, err: f64, delta2: f64, success: bool) {
        self.errs.push(err);
        self.deltas2.push(delta2);
        self.successes.push(success);
    }

    /// Whether no iteration was recorded.
    pub fn is_empty(&self) -> bool {
        self.deltas2.is_empty()
    }

    /// Squared norm of the last recorded step.
    pub fn last_delta2(&self) -> f64 {
        self.deltas2.last().copied().unwrap_or(0.0)
    }

    /// Number of recorded iterations.
    pub fn len(&self) -> usize {
        self.deltas2.len()
    }

    /// Remove all records.
    pub fn clear(&mut self) {
        self.errs.clear();
        self.deltas2.clear();
        self.successes.clear();
    }
}

/// Optimization results.
#[derive(Clone, Debug)]
pub struct Summary {
    /// Last valid cost.
    pub final_cost: Cost,
    /// Final relative error decrease.
    pub final_rerr_dec: f64,
    /// Final relative squared step norm.
    pub final_step_rel_norm2: f64,
    /// Why the optimization stopped.
    pub stop_reason: StopReason,
    /// Final number of residuals.
    pub num_residuals: u16,
    /// Number of successful iterations.
    pub num_iters: u16,
    /// Total number of failures to decrease the error.
    pub num_failures: u8,
    /// Number of consecutive failures at the end.
    pub num_consec_failures: u8,
    /// Cumulated optimization duration in milliseconds.
    pub duration_ms: f32,
    /// Final covariance matrix (single precision), when exported.
    pub final_cov: Option<crate::covariance::FinalCovariance>,
    /// Whether numerical derivatives were used.
    pub num_diff_used: bool,
    /// Per-iteration results.
    pub hist: History,
}

impl Default for Summary {
    fn default() -> Self {
        Self {
            final_cost: Cost::max(),
            final_rerr_dec: f64::MAX,
            final_step_rel_norm2: f64::MAX,
            stop_reason: StopReason::None,
            num_residuals: 0,
            num_iters: 0,
            num_failures: 0,
            num_consec_failures: 0,
            duration_ms: 0.0,
            final_cov: None,
            num_diff_used: false,
            hist: History::default(),
        }
    }
}

impl Summary {
    /// True when the stop reason is not a failure.
    pub fn succeeded(&self) -> bool {
        !self.stop_reason.is_failure()
    }

    /// True when the optimization reached a convergence criterion.
    pub fn converged(&self) -> bool {
        self.stop_reason >= StopReason::MinError && self.stop_reason < StopReason::MaxIters
    }

    /// Whether a final covariance was exported.
    pub fn has_final_cov(&self) -> bool {
        self.final_cov.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_and_convergence() {
        let mut s = Summary::default();
        assert!(s.succeeded());
        assert!(!s.converged());
        s.stop_reason = StopReason::MinError;
        assert!(s.converged());
        s.stop_reason = StopReason::MaxIters;
        assert!(s.succeeded());
        assert!(!s.converged());
        s.stop_reason = StopReason::SolverFailed;
        assert!(!s.succeeded());
    }

    #[test]
    fn history_records() {
        let mut h = History::default();
        assert!(h.is_empty());
        h.add(1.0, 0.25, true);
        h.add(0.5, 0.1, true);
        assert_eq!(h.len(), 2);
        assert_eq!(h.last_delta2(), 0.1);
        assert!(!h.is_empty());
    }
}
