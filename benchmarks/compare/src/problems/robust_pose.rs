// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Robust SE(3) pose problem, ported from `benchmarks/common/robust_pose.h`.
//!
//! The pose is a unit quaternion + translation with a right (`T * exp(ξ)`)
//! update, matching Sophus SE(3) used by the C++ tinyopt benchmark.
//!
//! The observation noise uses a small deterministic PRNG, so the Rust data is
//! not bit-identical to the C++ `std::mt19937` stream; sizes, outlier ratios
//! and noise levels are identical and the report states the difference.

/// Unit quaternion `(w, x, y, z)` and translation.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub q: [f64; 4],
    pub t: [f64; 3],
}

impl Pose {
    /// Rotation matrix (row-major).
    pub fn rotation(&self) -> [[f64; 3]; 3] {
        let [w, x, y, z] = self.q;
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }

    pub fn rotate(&self, p: &[f64; 3]) -> [f64; 3] {
        let r = self.rotation();
        [
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2],
        ]
    }

    pub fn transform(&self, p: &[f64; 3]) -> [f64; 3] {
        let r = self.rotate(p);
        [r[0] + self.t[0], r[1] + self.t[1], r[2] + self.t[2]]
    }

    /// `self * other`.
    pub fn compose(&self, other: &Pose) -> Pose {
        let r = self.rotation();
        let t = [
            r[0][0] * other.t[0] + r[0][1] * other.t[1] + r[0][2] * other.t[2] + self.t[0],
            r[1][0] * other.t[0] + r[1][1] * other.t[1] + r[1][2] * other.t[2] + self.t[1],
            r[2][0] * other.t[0] + r[2][1] * other.t[1] + r[2][2] * other.t[2] + self.t[2],
        ];
        let q = quat_mul(&self.q, &other.q);
        Pose {
            q: quat_normalize(q),
            t,
        }
    }

    /// Right update `T <- T * exp(ξ)`.
    pub fn plus(&self, xi: &[f64; 6]) -> Pose {
        self.compose(&Pose::exp(xi))
    }

    /// Flat parameters `(qx, qy, qz, qw, tx, ty, tz)`? No: `(w, x, y, z, t…)`.
    pub fn params(&self) -> [f64; 7] {
        [
            self.q[0], self.q[1], self.q[2], self.q[3], self.t[0], self.t[1], self.t[2],
        ]
    }

    /// SE(3) exponential map from `(ω, ν)`.
    pub fn exp(xi: &[f64; 6]) -> Pose {
        let w = [xi[0], xi[1], xi[2]];
        let v = [xi[3], xi[4], xi[5]];
        let theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
        let theta = theta2.sqrt();
        let (a, b, c) = if theta < 1e-8 {
            (
                1.0 - theta2 / 6.0,
                0.5 - theta2 / 24.0,
                1.0 / 6.0 - theta2 / 120.0,
            )
        } else {
            let s = theta.sin();
            let co = theta.cos();
            (
                s / theta,
                (1.0 - co) / theta2,
                (theta - s) / (theta2 * theta),
            )
        };
        let k = skew(&w);
        let k2 = mul3(k, k);
        let rot = add3(add3(identity3(), scale3(k, a)), scale3(k2, b));
        let vel = add3(add3(identity3(), scale3(k, b)), scale3(k2, c));
        let t = [
            vel[0][0] * v[0] + vel[0][1] * v[1] + vel[0][2] * v[2],
            vel[1][0] * v[0] + vel[1][1] * v[1] + vel[1][2] * v[2],
            vel[2][0] * v[0] + vel[2][1] * v[1] + vel[2][2] * v[2],
        ];
        Pose {
            q: quat_from_rotation(&rot),
            t,
        }
    }
}

/// Residual Jacobian w.r.t. the right update `(ω, ν)` (3x6, row-major).
pub fn pose_jacobian(pose: &Pose, point: &[f64; 3]) -> [f64; 18] {
    let r = pose.rotation();
    let mut j = [0.0; 18];
    for row in 0..3 {
        // -R [p]x
        j[row * 6] = -(r[row][1] * point[2] - r[row][2] * point[1]);
        j[row * 6 + 1] = -(r[row][2] * point[0] - r[row][0] * point[2]);
        j[row * 6 + 2] = -(r[row][0] * point[1] - r[row][1] * point[0]);
        // R
        j[row * 6 + 3] = r[row][0];
        j[row * 6 + 4] = r[row][1];
        j[row * 6 + 5] = r[row][2];
    }
    j
}

pub struct Observation {
    pub world_point: [f64; 3],
    pub measurement: [f64; 3],
}

pub fn ground_truth() -> Pose {
    Pose::exp(&[0.15, -0.1, 0.2, 0.3, -0.2, 0.1])
}

pub fn initial_pose(ground_truth: &Pose) -> Pose {
    ground_truth.plus(&[0.05, -0.04, 0.06, 0.07, -0.06, 0.04])
}

/// Deterministic observations: `outliers` of them carry σ=50 noise.
pub fn make_observations(
    ground_truth: &Pose,
    inliers: usize,
    outliers: usize,
    inlier_sigma: f64,
    outlier_sigma: f64,
) -> Vec<Observation> {
    let mut rng = Rng::new(42);
    let mut observations = Vec::with_capacity(inliers + outliers);
    for index in 0..inliers + outliers {
        let sigma = if index < inliers {
            inlier_sigma
        } else {
            outlier_sigma
        };
        let point = [
            rng.uniform(-5.0, 5.0),
            rng.uniform(-5.0, 5.0),
            rng.uniform(-5.0, 5.0),
        ];
        let transformed = ground_truth.transform(&point);
        observations.push(Observation {
            world_point: point,
            measurement: [
                transformed[0] + rng.normal() * sigma,
                transformed[1] + rng.normal() * sigma,
                transformed[2] + rng.normal() * sigma,
            ],
        });
    }
    observations
}

// --- Small deterministic PRNG (PCG32 + Box-Muller) --------------------------

struct Rng {
    state: u64,
    spare: Option<f64>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
            spare: None,
        }
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        let u = self.next_u32() as f64 / (u32::MAX as f64 + 1.0);
        low + (high - low) * u
    }

    fn normal(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        loop {
            let u = 2.0 * self.uniform(0.0, 1.0) - 1.0;
            let v = 2.0 * self.uniform(0.0, 1.0) - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let factor = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * factor);
                return u * factor;
            }
        }
    }
}

// --- Tiny fixed-size matrix helpers -----------------------------------------

fn identity3() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn skew(w: &[f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
}

fn mul3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn scale3(a: [[f64; 3]; 3], s: f64) -> [[f64; 3]; 3] {
    let mut out = a;
    for row in &mut out {
        for value in row {
            *value *= s;
        }
    }
    out
}

fn add3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = a;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] += b[i][j];
        }
    }
    out
}

fn quat_mul(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
}

fn quat_from_rotation(r: &[[f64; 3]; 3]) -> [f64; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        [
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        ]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        [
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        ]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        [
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        ]
    };
    quat_normalize(q)
}
