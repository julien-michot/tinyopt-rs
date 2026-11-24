// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Cost value and residual bookkeeping, mirroring tinyopt's `Cost`.

use core::fmt;

/// A cost value together with residual statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    /// The function cost (usually the squared norm of the residuals).
    pub cost: f64,
    /// The number of residuals (for NLLS problems).
    pub num_residuals: i32,
    /// The ratio of inlier residuals (when robust losses are used).
    pub inlier_ratio: f32,
}

impl Default for Cost {
    fn default() -> Self {
        Self {
            cost: 0.0,
            num_residuals: 0,
            inlier_ratio: 1.0,
        }
    }
}

impl Cost {
    /// A scalar cost counted as a single residual.
    pub fn new(cost: f64) -> Self {
        Self {
            cost,
            num_residuals: 1,
            inlier_ratio: 1.0,
        }
    }

    /// A cost with an explicit number of residuals.
    pub fn with_num_residuals(cost: f64, num_residuals: i32) -> Self {
        Self {
            cost,
            num_residuals,
            inlier_ratio: 1.0,
        }
    }

    /// A cost with an explicit number of residuals and inlier ratio.
    pub fn with_inliers(cost: f64, num_residuals: i32, inlier_ratio: f32) -> Self {
        Self {
            cost,
            num_residuals,
            inlier_ratio,
        }
    }

    /// Cost and residual count from a residual slice (squared L2 norm).
    pub fn from_residuals(residuals: &[f64]) -> Self {
        Self::with_num_residuals(
            residuals.iter().map(|r| r * r).sum(),
            residuals.len() as i32,
        )
    }

    /// The largest representable cost, used as the initial best value.
    pub fn max() -> Self {
        Self {
            cost: f64::MAX,
            num_residuals: 0,
            inlier_ratio: 1.0,
        }
    }

    /// A valid cost has residuals and is not the maximum sentinel.
    pub fn is_valid(&self) -> bool {
        self.num_residuals > 0 && self.cost != f64::MAX
    }

    /// Number of residuals considered inliers.
    pub fn num_inliers(&self) -> i32 {
        (self.num_residuals as f32 * self.inlier_ratio) as i32
    }

    /// Number of residuals considered outliers.
    pub fn num_outliers(&self) -> i32 {
        (self.num_residuals as f32 * (1.0 - self.inlier_ratio)) as i32
    }

    /// Accumulate another cost and update the inlier ratio.
    pub fn add_residuals(&mut self, new_residuals: i32, new_inliers: i32) {
        let total = self.num_residuals + new_residuals;
        if total > 0 {
            self.inlier_ratio = (self.num_inliers() + new_inliers) as f32 / total as f32;
        }
        self.num_residuals = total;
    }
}

impl core::ops::AddAssign for Cost {
    fn add_assign(&mut self, other: Self) {
        self.cost += other.cost;
        self.add_residuals(other.num_residuals, other.num_inliers());
    }
}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.cost.partial_cmp(&other.cost)
    }
}

impl fmt::Display for Cost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ε:{:.4e}, n:{}", self.cost, self.num_residuals)?;
        if self.num_residuals > 1 {
            write!(
                f,
                ", √ε/n:{:.2e}",
                (self.cost / self.num_residuals as f64).sqrt()
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_based_cost() {
        let c = Cost::from_residuals(&[1.0, 2.0, 2.0]);
        assert_eq!(c.cost, 9.0);
        assert_eq!(c.num_residuals, 3);
        assert_eq!(c.num_inliers(), 3);
        assert!(c.is_valid());
    }

    #[test]
    fn accumulate_costs() {
        let mut a = Cost::with_inliers(1.0, 2, 0.5);
        let b = Cost::with_inliers(4.0, 2, 1.0);
        a += b;
        assert_eq!(a.cost, 5.0);
        assert_eq!(a.num_residuals, 4);
        assert_eq!(a.num_inliers(), 3);
    }

    #[test]
    fn sentinel_is_invalid() {
        assert!(!Cost::max().is_valid());
        assert!(!Cost::default().is_valid());
    }
}
