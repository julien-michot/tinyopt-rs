// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Termination reasons, mirroring tinyopt's `StopReason` ordering.

use crate::options::Options;
use core::fmt;

/// Why the optimization terminated.
///
/// The declaration order matters: failure variants sort before [`StopReason::None`]
/// and the success/convergence variants after it, exactly like the C++ enum values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum StopReason {
    /// Out of memory when allocating the system.
    OutOfMemory,
    /// Failed to solve the normal equations.
    SolverFailed,
    /// Residuals or Jacobians have NaNs or infinities.
    SystemHasNanOrInf,
    /// Nothing to optimize, or the system has no residuals.
    Skipped,

    /// No stop, used by single steps or when no iteration ran (success).
    #[default]
    None,
    /// Minimal error reached (success).
    MinError,
    /// Minimal relative error decrease reached (success).
    MinRelError,
    /// Minimal step norm reached (success).
    MinDeltaNorm,
    /// Minimal relative step norm reached (success).
    MinRelDeltaNorm,
    /// Minimal gradient reached (success).
    MinGradNorm,
    /// Maximum number of successful iterations reached (success).
    MaxIters,
    /// Failed to decrease the error too many times (success).
    MaxNoDecr,
    /// Failed to decrease the error consecutively too many times (success).
    MaxConsecNoDecr,
    /// Maximum allocated time reached (success).
    TimedOut,
    /// User stopped the process (success).
    UserStopped,
}

impl StopReason {
    /// Whether this reason denotes a failure.
    pub fn is_failure(self) -> bool {
        self < StopReason::None
    }

    /// Human-readable description, optionally including the matching option value.
    pub fn description(self, options: Option<&Options>, final_cost: f64) -> String {
        let mut s = String::from(match self {
            StopReason::None => "🌱 Optimization not ran or used with a single step (success)",
            StopReason::MinError => "🌞 Reached minimum error (success)",
            StopReason::MinRelError => "🌞 Reached minimum relative error (success)",
            StopReason::MinDeltaNorm => "🌞 Reached minimal delta norm (success)",
            StopReason::MinRelDeltaNorm => "🌞 Reached minimal relative delta norm (success)",
            StopReason::MinGradNorm => "🌞 Reached minimal gradient (success)",
            StopReason::MaxIters => "⛅ Reached maximum number of successful iterations (success)",
            StopReason::MaxNoDecr => "⛅ Failed to decrease error too many times (success)",
            StopReason::MaxConsecNoDecr => {
                "⛅ Failed to decrease error consecutively too many times (success)"
            }
            StopReason::TimedOut => "⌛ Reached maximum allocated time (success)",
            StopReason::UserStopped => "👍 User stopped the process (success)",
            StopReason::OutOfMemory => "❌ Out of memory when allocating the Hessian(s) (failure)",
            StopReason::SystemHasNanOrInf => "❌ Residuals or Jacobians have NaNs or Inf (failure)",
            StopReason::SolverFailed => "❌ Failed to solve the normal equations (failure)",
            StopReason::Skipped => {
                "❌ The system has no residuals or nothing to optimize (failure)"
            }
        });
        if let Some(o) = options {
            match self {
                StopReason::MinError => {
                    s.push_str(&format!(" ε:[{final_cost:.3e} < {:.3e}]", o.stop.min_error));
                }
                StopReason::MinRelError => {
                    s.push_str(&format!(" rerr:[< {:.3e}]", o.stop.min_rerr_dec));
                }
                StopReason::MinGradNorm => {
                    if o.stop.min_grad_max > 0.0 {
                        s.push_str(&format!(" |∇|∞:[< {:.3e}]", o.stop.min_grad_max));
                    } else {
                        s.push_str(&format!(" |∇|:[< {:.3e}]", o.stop.min_grad_norm2.sqrt()));
                    }
                }
                StopReason::MaxIters => {
                    s.push_str(&format!(" [#it == {}]", o.stop.max_iters));
                }
                StopReason::TimedOut => {
                    s.push_str(&format!(" τ:[> {}ms]", o.stop.max_duration_ms));
                }
                _ => {}
            }
        }
        s
    }
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.description(None, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_matches_cpp() {
        assert!(StopReason::OutOfMemory < StopReason::SolverFailed);
        assert!(StopReason::Skipped < StopReason::None);
        assert!(StopReason::MinError < StopReason::MaxIters);
        assert!(!StopReason::None.is_failure());
        assert!(StopReason::SolverFailed.is_failure());
        assert!(StopReason::Skipped.is_failure());
    }

    #[test]
    fn descriptions_include_values() {
        let o = Options::default();
        let s = StopReason::MinError.description(Some(&o), 1e-13);
        assert!(s.contains("1.000e-13"));
    }
}
