// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Bundle-adjustment generator, ported from `benchmarks/common/bundle_adjustment.h`.

pub const FOCAL_X: f64 = 520.0;
pub const FOCAL_Y: f64 = 520.0;
pub const PRINCIPAL_X: f64 = 320.0;
pub const PRINCIPAL_Y: f64 = 240.0;
pub const POINT_CAMERA_WINDOW: usize = 5;

/// Camera pose `(w, x, y, z, tx, ty, tz)` in the SE(3) representation.
pub type Camera = [f64; 7];
/// Landmark position.
pub type Point = [f64; 3];
/// Image measurement `[u, v]`.
pub type ImagePoint = [f64; 2];

pub struct Observation {
    pub camera: usize,
    pub point: usize,
    pub measurement: ImagePoint,
}

pub struct Problem {
    pub truth_cameras: Vec<Camera>,
    pub initial_cameras: Vec<Camera>,
    pub truth_points: Vec<Point>,
    pub initial_points: Vec<Point>,
    pub observations: Vec<Observation>,
}

impl Problem {
    pub fn camera_count(&self) -> usize {
        self.truth_cameras.len()
    }

    pub fn point_count(&self) -> usize {
        self.truth_points.len()
    }
}

fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
}

fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn rotation_matrix(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [w, x, y, z] = quat_normalize(q);
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

fn quat_from_rotation(r: [[f64; 3]; 3]) -> [f64; 4] {
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

fn skew(v: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

fn mul3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn add3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

fn scale3(a: [[f64; 3]; 3], s: f64) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] * s;
        }
    }
    out
}

fn identity3() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

pub fn se3_exp(xi: &[f64; 6]) -> Camera {
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
    let k = skew(w);
    let k2 = mul3(k, k);
    let r = add3(add3(identity3(), scale3(k, a)), scale3(k2, b));
    let vel = add3(add3(identity3(), scale3(k, b)), scale3(k2, c));
    let t = [
        vel[0][0] * v[0] + vel[0][1] * v[1] + vel[0][2] * v[2],
        vel[1][0] * v[0] + vel[1][1] * v[1] + vel[1][2] * v[2],
        vel[2][0] * v[0] + vel[2][1] * v[1] + vel[2][2] * v[2],
    ];
    let q = quat_from_rotation(r);
    [q[0], q[1], q[2], q[3], t[0], t[1], t[2]]
}

pub fn from_rotation_vector(rvec: [f64; 3], translation: [f64; 3]) -> Camera {
    let theta = (rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]).sqrt();
    let q = if theta < 1e-12 {
        [1.0, rvec[0] * 0.5, rvec[1] * 0.5, rvec[2] * 0.5]
    } else {
        let half = theta * 0.5;
        let sin_half = half.sin();
        let cos_half = half.cos();
        let scale = sin_half / theta;
        [cos_half, rvec[0] * scale, rvec[1] * scale, rvec[2] * scale]
    };
    [
        q[0],
        q[1],
        q[2],
        q[3],
        translation[0],
        translation[1],
        translation[2],
    ]
}

pub fn se3_plus(camera: &Camera, xi: &[f64; 6]) -> Camera {
    let inc = se3_exp(xi);
    let q = quat_mul(
        [camera[0], camera[1], camera[2], camera[3]],
        [inc[0], inc[1], inc[2], inc[3]],
    );
    let r = rotation_matrix([camera[0], camera[1], camera[2], camera[3]]);
    let inc_t = [inc[4], inc[5], inc[6]];
    let t = [
        r[0][0] * inc_t[0] + r[0][1] * inc_t[1] + r[0][2] * inc_t[2] + camera[4],
        r[1][0] * inc_t[0] + r[1][1] * inc_t[1] + r[1][2] * inc_t[2] + camera[5],
        r[2][0] * inc_t[0] + r[2][1] * inc_t[1] + r[2][2] * inc_t[2] + camera[6],
    ];
    let q = quat_normalize(q);
    [q[0], q[1], q[2], q[3], t[0], t[1], t[2]]
}

/// Pinhole projection of a world point through an SE(3) camera.
pub fn project(camera: &Camera, point: &Point) -> ImagePoint {
    let r = rotation_matrix([camera[0], camera[1], camera[2], camera[3]]);
    let x = r[0][0] * point[0] + r[0][1] * point[1] + r[0][2] * point[2] + camera[4];
    let y = r[1][0] * point[0] + r[1][1] * point[1] + r[1][2] * point[2] + camera[5];
    let z = r[2][0] * point[0] + r[2][1] * point[1] + r[2][2] * point[2] + camera[6];
    [FOCAL_X * x / z + PRINCIPAL_X, FOCAL_Y * y / z + PRINCIPAL_Y]
}

/// Projection and its Jacobians: `(2x6 camera, 2x3 point)`, row-major.
pub fn projection_jacobians(camera: &Camera, point: &Point) -> ([f64; 12], [f64; 6]) {
    let r = rotation_matrix([camera[0], camera[1], camera[2], camera[3]]);
    let x = r[0][0] * point[0] + r[0][1] * point[1] + r[0][2] * point[2] + camera[4];
    let y = r[1][0] * point[0] + r[1][1] * point[1] + r[1][2] * point[2] + camera[5];
    let z = r[2][0] * point[0] + r[2][1] * point[1] + r[2][2] * point[2] + camera[6];
    let projection = [
        [FOCAL_X / z, 0.0, -FOCAL_X * x / (z * z)],
        [0.0, FOCAL_Y / z, -FOCAL_Y * y / (z * z)],
    ];

    let neg_r_skew = [
        [
            -r[0][1] * point[2] + r[0][2] * point[1],
            r[0][0] * point[2] - r[0][2] * point[0],
            -r[0][0] * point[1] + r[0][1] * point[0],
        ],
        [
            -r[1][1] * point[2] + r[1][2] * point[1],
            r[1][0] * point[2] - r[1][2] * point[0],
            -r[1][0] * point[1] + r[1][1] * point[0],
        ],
        [
            -r[2][1] * point[2] + r[2][2] * point[1],
            r[2][0] * point[2] - r[2][2] * point[0],
            -r[2][0] * point[1] + r[2][1] * point[0],
        ],
    ];

    let mut camera_jacobian = [0.0; 12];
    for row in 0..2 {
        for col in 0..3 {
            camera_jacobian[row * 6 + col] = projection[row][0] * neg_r_skew[0][col]
                + projection[row][1] * neg_r_skew[1][col]
                + projection[row][2] * neg_r_skew[2][col];
            camera_jacobian[row * 6 + 3 + col] = projection[row][0] * r[0][col]
                + projection[row][1] * r[1][col]
                + projection[row][2] * r[2][col];
        }
    }
    let mut point_jacobian = [0.0; 6];
    for row in 0..2 {
        for col in 0..3 {
            point_jacobian[row * 3 + col] = projection[row][0] * r[0][col]
                + projection[row][1] * r[1][col]
                + projection[row][2] * r[2][col];
        }
    }
    (camera_jacobian, point_jacobian)
}

/// Projection Jacobians for an ambient quaternion + translation camera block.
/// The camera Jacobian is row-major `2x7` and includes quaternion normalization.
#[cfg(any(feature = "ceres", test))]
pub fn projection_jacobians_ambient(camera: &Camera, point: &Point) -> ([f64; 14], [f64; 6]) {
    let raw_q = [camera[0], camera[1], camera[2], camera[3]];
    let norm = raw_q.iter().map(|value| value * value).sum::<f64>().sqrt();
    let q = quat_normalize(raw_q);
    let rotation = rotation_matrix(q);
    let rotated = [
        rotation[0][0] * point[0] + rotation[0][1] * point[1] + rotation[0][2] * point[2],
        rotation[1][0] * point[0] + rotation[1][1] * point[1] + rotation[1][2] * point[2],
        rotation[2][0] * point[0] + rotation[2][1] * point[1] + rotation[2][2] * point[2],
    ];
    let x = rotated[0] + camera[4];
    let y = rotated[1] + camera[5];
    let z = rotated[2] + camera[6];
    let projection = [
        [FOCAL_X / z, 0.0, -FOCAL_X * x / (z * z)],
        [0.0, FOCAL_Y / z, -FOCAL_Y * y / (z * z)],
    ];

    let rotation_jacobian = quaternion_rotation_jacobian(&q, point);
    let mut camera_jacobian = [0.0; 14];
    for row in 0..2 {
        for col in 0..4 {
            for axis in 0..4 {
                let identity = if axis == col { 1.0 } else { 0.0 };
                let normalization = (identity - q[axis] * q[col]) / norm;
                let mut derivative = 0.0;
                for coordinate in 0..3 {
                    derivative +=
                        projection[row][coordinate] * rotation_jacobian[coordinate * 4 + axis];
                }
                camera_jacobian[row * 7 + col] += derivative * normalization;
            }
        }
        for col in 0..3 {
            camera_jacobian[row * 7 + 4 + col] = projection[row][col];
        }
    }
    let (_, point_jacobian) = projection_jacobians(camera, point);
    (camera_jacobian, point_jacobian)
}

/// Derivative of `R(q) * point` with respect to the unit quaternion `(w,x,y,z)`.
#[cfg(any(feature = "ceres", test))]
pub fn quaternion_rotation_jacobian(q: &[f64; 4], point: &Point) -> [f64; 12] {
    let (w, v) = (q[0], [q[1], q[2], q[3]]);
    let dot = v[0] * point[0] + v[1] * point[1] + v[2] * point[2];
    let cross = [
        v[1] * point[2] - v[2] * point[1],
        v[2] * point[0] - v[0] * point[2],
        v[0] * point[1] - v[1] * point[0],
    ];
    let skew = [
        [0.0, -point[2], point[1]],
        [point[2], 0.0, -point[0]],
        [-point[1], point[0], 0.0],
    ];
    let mut jacobian = [0.0; 12];
    for row in 0..3 {
        jacobian[row * 4] = 2.0 * w * point[row] + 2.0 * cross[row];
        for col in 0..3 {
            jacobian[row * 4 + col + 1] = -2.0 * point[row] * v[col]
                + 2.0 * dot * if row == col { 1.0 } else { 0.0 }
                + 2.0 * v[row] * point[col]
                - 2.0 * w * skew[row][col];
        }
    }
    jacobian
}

/// Deterministic problem generator (noise-free, identical to the C++ one).
pub fn make_problem(camera_count: usize, point_count: usize) -> Problem {
    assert!(camera_count >= 2 && point_count >= 2);
    let mut problem = Problem {
        truth_cameras: Vec::with_capacity(camera_count),
        initial_cameras: Vec::with_capacity(camera_count),
        truth_points: Vec::with_capacity(point_count),
        initial_points: Vec::with_capacity(point_count),
        observations: Vec::new(),
    };
    let half = (camera_count as f64 - 1.0) * 0.5;
    for index in 0..camera_count {
        let i = index as f64;
        let camera = from_rotation_vector(
            [0.002 * i, -0.018 * (i - half), 0.003 * i],
            [-0.18 * (i - half), 0.015 * (i - half), 0.02 * (i - half)],
        );
        problem.truth_cameras.push(camera);
        let initial = if index != 0 {
            se3_plus(&camera, &[0.006 * i, -0.004, 0.0, 0.035, -0.02 * i, 0.025])
        } else {
            camera
        };
        problem.initial_cameras.push(initial);
    }

    for index in 0..point_count {
        let point = [
            (index % 10) as f64 * 0.24 - 4.5 * 0.24,
            ((index / 10) % 20) as f64 * 0.12 - 9.5 * 0.12,
            4.5 + 0.11 * (index % 7) as f64,
        ];
        problem.truth_points.push(point);
        let mut initial = point;
        if index != 0 {
            initial[0] += 0.02;
            initial[1] -= 0.015;
            initial[2] += 0.025;
        }
        problem.initial_points.push(initial);
    }

    let window = camera_count.min(POINT_CAMERA_WINDOW);
    let start_count = camera_count - window + 1;
    for point_index in 0..point_count {
        let first_camera = point_index % start_count;
        for offset in 0..window {
            let camera_index = first_camera + offset;
            let measurement = project(
                &problem.truth_cameras[camera_index],
                &problem.truth_points[point_index],
            );
            problem.observations.push(Observation {
                camera: camera_index,
                point: point_index,
                measurement,
            });
        }
    }
    problem
}

/// Initial reprojection cost `0.5 Σ ‖r‖²`.
pub fn initial_cost(problem: &Problem) -> f64 {
    let mut squared_error = 0.0;
    for observation in &problem.observations {
        let projected = project(
            &problem.initial_cameras[observation.camera],
            &problem.initial_points[observation.point],
        );
        let dx = projected[0] - observation.measurement[0];
        let dy = projected[1] - observation.measurement[1];
        squared_error += dx * dx + dy * dy;
    }
    0.5 * squared_error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambient_se3_projection_jacobians_match_finite_differences() {
        let camera = from_rotation_vector([0.13, -0.07, 0.21], [0.4, -0.2, 0.8]);
        let point = [0.3, -0.6, 5.2];
        let (camera_jacobian, point_jacobian) = projection_jacobians_ambient(&camera, &point);
        let epsilon = 1e-7;
        let mut max_error = 0.0_f64;
        let mut max_coordinate = (0, 0);
        let mut scale = 1.0_f64;

        for col in 0..7 {
            let mut plus = camera;
            let mut minus = camera;
            plus[col] += epsilon;
            minus[col] -= epsilon;
            let projected_plus = project(&plus, &point);
            let projected_minus = project(&minus, &point);
            for row in 0..2 {
                let numerical = (projected_plus[row] - projected_minus[row]) / (2.0 * epsilon);
                let analytical = camera_jacobian[row * 7 + col];
                if (numerical - analytical).abs() > max_error {
                    max_error = (numerical - analytical).abs();
                    max_coordinate = (row, col);
                }
                scale = scale.max(analytical.abs());
            }
        }
        for col in 0..3 {
            let mut plus = point;
            let mut minus = point;
            plus[col] += epsilon;
            minus[col] -= epsilon;
            let projected_plus = project(&camera, &plus);
            let projected_minus = project(&camera, &minus);
            for row in 0..2 {
                let numerical = (projected_plus[row] - projected_minus[row]) / (2.0 * epsilon);
                let analytical = point_jacobian[row * 3 + col];
                if (numerical - analytical).abs() > max_error {
                    max_error = (numerical - analytical).abs();
                    max_coordinate = (row, col + 7);
                }
                scale = scale.max(analytical.abs());
            }
        }

        assert!(
            max_error < 1e-5 * scale,
            "ambient SE(3) Jacobian max error: {max_error} at {max_coordinate:?}"
        );
    }
}
