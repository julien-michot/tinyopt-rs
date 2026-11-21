// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Problem definitions shared by every backend, ported from the C++ suite.

pub mod bundle_adjustment;
pub mod dense;
pub mod robust_pose;
pub mod sparse;

/// Which family a benchmark case belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Dense math problem with a static parameter vector.
    DenseMathStatic { dims: usize, single: bool },
    /// Dense math problem with a dynamic parameter vector.
    DenseMathDynamic { dims: usize, single: bool },
    /// Dense prior (diagonal) problem.
    DensePrior { dims: usize },
    /// Scalar sparse chain.
    SparseChain { dims: usize },
    /// Bundle adjustment with a block-sparse Hessian.
    BundleAdjustmentBlock { cams: usize, points: usize },
    /// Bundle adjustment with a scalar sparse Hessian.
    BundleAdjustmentSparse { cams: usize, points: usize },
    /// Robust pose with (optionally) a Huber loss.
    RobustPose { obs: usize, robust: bool },
}

impl Kind {
    /// Report category, matching the C++ labels.
    pub fn category(&self) -> &'static str {
        match self {
            Kind::DenseMathStatic { .. } => "Dense static",
            Kind::DenseMathDynamic { .. } => "Dense dynamic",
            Kind::DensePrior { .. } => "Dense dynamic",
            Kind::SparseChain { .. } => "Sparse",
            Kind::BundleAdjustmentBlock { .. } | Kind::BundleAdjustmentSparse { .. } => {
                "Bundle adjustment"
            }
            Kind::RobustPose { .. } => "Robust pose",
        }
    }
}

/// One benchmark case.
#[derive(Clone, Debug)]
pub struct Case {
    pub kind: Kind,
    /// C++ benchmark label, e.g. `"50D dynamic double prior"` or `"2c 200p sparse"`.
    pub name: String,
    /// Short problem label used by the iteration table, e.g. `"50dp"` or `"2c 200p"`.
    pub short: String,
}

/// All cases of the C++ suite.
///
/// `full` adds the very large bundle-adjustment case (100 cameras, 50k points);
/// `quick` keeps one representative case per family.
pub fn cases(full: bool, quick: bool) -> Vec<Case> {
    let case = |kind: Kind, name: String, short: String| Case { kind, name, short };
    let mut out = Vec::new();
    if quick {
        out.push(case(
            Kind::DenseMathStatic {
                dims: 3,
                single: false,
            },
            "3D static double".into(),
            "3d".into(),
        ));
        out.push(case(
            Kind::DensePrior { dims: 33 },
            "33D dynamic double prior".into(),
            "33dp".into(),
        ));
        out.push(case(
            Kind::SparseChain { dims: 100 },
            "100D sparse chain".into(),
            "100d".into(),
        ));
        out.push(case(
            Kind::BundleAdjustmentBlock {
                cams: 2,
                points: 200,
            },
            "2c 200p".into(),
            "2c 200p".into(),
        ));
        out.push(case(
            Kind::RobustPose {
                obs: 50,
                robust: true,
            },
            "50 obs Huber".into(),
            "50o Huber".into(),
        ));
        return out;
    }

    for dims in 1..=3 {
        out.push(case(
            Kind::DenseMathStatic {
                dims,
                single: false,
            },
            format!("{dims}D static double"),
            format!("{dims}d"),
        ));
        out.push(case(
            Kind::DenseMathStatic { dims, single: true },
            format!("{dims}D static float"),
            format!("{dims}f"),
        ));
    }
    for dims in 1..=3 {
        out.push(case(
            Kind::DenseMathDynamic {
                dims,
                single: false,
            },
            format!("{dims}D dynamic double"),
            format!("{dims}d"),
        ));
        out.push(case(
            Kind::DenseMathDynamic { dims, single: true },
            format!("{dims}D dynamic float"),
            format!("{dims}f"),
        ));
    }
    for dims in [6usize, 12, 33, 50] {
        out.push(case(
            Kind::DensePrior { dims },
            format!("{dims}D dynamic double prior"),
            format!("{dims}dp"),
        ));
    }
    for dims in [10usize, 100, 1000] {
        out.push(case(
            Kind::SparseChain { dims },
            format!("{dims}D sparse chain"),
            format!("{dims}d"),
        ));
    }
    for (cams, points) in [(2usize, 200usize), (4, 600), (10, 3000)] {
        out.push(case(
            Kind::BundleAdjustmentBlock { cams, points },
            format!("{cams}c {points}p"),
            format!("{cams}c {points}p"),
        ));
        out.push(case(
            Kind::BundleAdjustmentSparse { cams, points },
            format!("{cams}c {points}p sparse"),
            format!("{cams}c {points}p sparse"),
        ));
    }
    if full {
        out.push(case(
            Kind::BundleAdjustmentBlock {
                cams: 100,
                points: 50_000,
            },
            "100c 50000p".into(),
            "100c 50000p".into(),
        ));
        out.push(case(
            Kind::BundleAdjustmentSparse {
                cams: 100,
                points: 50_000,
            },
            "100c 50000p sparse".into(),
            "100c 50000p sparse".into(),
        ));
    }
    for obs in [20usize, 50, 100] {
        for robust in [false, true] {
            out.push(case(
                Kind::RobustPose { obs, robust },
                format!("{obs} obs {}", if robust { "Huber" } else { "L2" }),
                format!("{obs}o {}", if robust { "Huber" } else { "L2" }),
            ));
        }
    }
    out
}
