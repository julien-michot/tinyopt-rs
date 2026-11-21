// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Bundle-adjustment generator, ported from `benchmarks/common/bundle_adjustment.h`.

pub const FOCAL_X: f64 = 520.0;
pub const FOCAL_Y: f64 = 520.0;
pub const PRINCIPAL_X: f64 = 320.0;
pub const PRINCIPAL_Y: f64 = 240.0;
pub const POINT_CAMERA_WINDOW: usize = 5;

/// Camera pose `[rx, ry, rz, tx, ty, tz]`.
pub type Camera = [f64; 6];
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

/// `Rz(rz) Ry(ry) Rx(rx) p`.
pub fn rotate_euler(angles: &[f64; 3], point: &[f64; 3]) -> [f64; 3] {
    let (cx, sx) = (angles[0].cos(), angles[0].sin());
    let (cy, sy) = (angles[1].cos(), angles[1].sin());
    let (cz, sz) = (angles[2].cos(), angles[2].sin());
    let y1 = cx * point[1] - sx * point[2];
    let z1 = sx * point[1] + cx * point[2];
    let x2 = cy * point[0] + sy * z1;
    let z2 = -sy * point[0] + cy * z1;
    [cz * x2 - sz * y1, sz * x2 + cz * y1, z2]
}

/// Pinhole projection of a world point through a camera.
pub fn project(camera: &Camera, point: &Point) -> ImagePoint {
    let angles = [camera[0], camera[1], camera[2]];
    let rotated = rotate_euler(&angles, point);
    let x = rotated[0] + camera[3];
    let y = rotated[1] + camera[4];
    let z = rotated[2] + camera[5];
    [FOCAL_X * x / z + PRINCIPAL_X, FOCAL_Y * y / z + PRINCIPAL_Y]
}

/// Projection and its Jacobians: `(2x6 camera, 2x3 point)`, row-major.
pub fn projection_jacobians(camera: &Camera, point: &Point) -> ([f64; 12], [f64; 6]) {
    let (cx, sx) = (camera[0].cos(), camera[0].sin());
    let (cy, sy) = (camera[1].cos(), camera[1].sin());
    let (cz, sz) = (camera[2].cos(), camera[2].sin());

    let rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
    let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
    let drx = [[0.0, 0.0, 0.0], [0.0, -sx, -cx], [0.0, cx, -sx]];
    let dry = [[-sy, 0.0, cy], [0.0, 0.0, 0.0], [-cy, 0.0, -sy]];
    let drz = [[-sz, -cz, 0.0], [cz, -sz, 0.0], [0.0, 0.0, 0.0]];
    let mul = |a: [[f64; 3]; 3], b: [[f64; 3]; 3]| {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
            }
        }
        out
    };
    let apply = |a: [[f64; 3]; 3], v: &[f64; 3]| {
        [
            a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
            a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
            a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
        ]
    };

    let ry_rx = mul(ry, rx);
    let rotation = mul(rz, ry_rx);
    let rotated = apply(rotation, point);
    let x = rotated[0] + camera[3];
    let y = rotated[1] + camera[4];
    let z = rotated[2] + camera[5];

    let projection = [
        [FOCAL_X / z, 0.0, -FOCAL_X * x / (z * z)],
        [0.0, FOCAL_Y / z, -FOCAL_Y * y / (z * z)],
    ];
    let rotation_jacobian = [
        apply(mul(rz, mul(ry, drx)), point),
        apply(mul(rz, mul(dry, rx)), point),
        apply(mul(drz, ry_rx), point),
    ];
    let mut camera_jacobian = [0.0; 12];
    for row in 0..2 {
        for col in 0..3 {
            camera_jacobian[row * 6 + col] = (0..3)
                .map(|k| projection[row][k] * rotation_jacobian[col][k])
                .sum();
            camera_jacobian[row * 6 + 3 + col] = projection[row][col];
        }
    }
    let mut point_jacobian = [0.0; 6];
    for row in 0..2 {
        for col in 0..3 {
            point_jacobian[row * 3 + col] =
                (0..3).map(|k| projection[row][k] * rotation[k][col]).sum();
        }
    }
    (camera_jacobian, point_jacobian)
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
        let mut camera = [0.0; 6];
        camera[0] = 0.002 * i;
        camera[1] = -0.018 * (i - half);
        camera[2] = 0.003 * i;
        camera[3] = -0.18 * (i - half);
        camera[4] = 0.015 * (i - half);
        camera[5] = 0.02 * (i - half);
        problem.truth_cameras.push(camera);
        let mut initial = camera;
        if index != 0 {
            initial[0] += 0.006 * i;
            initial[1] -= 0.004;
            initial[3] += 0.035;
            initial[4] -= 0.02 * i;
            initial[5] += 0.025;
        }
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
