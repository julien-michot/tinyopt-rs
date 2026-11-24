struct Params {
    camera_count: u32,
    point_count: u32,
    window: u32,
    dims: u32,
}

@group(0) @binding(0) var<storage, read> ambient: array<f32>;
@group(0) @binding(1) var<storage, read> observations: array<f32>;
@group(0) @binding(2) var<storage, read> ba_meta: Params;
@group(0) @binding(3) var<storage, read_write> cost: array<f32>;
@group(0) @binding(4) var<storage, read_write> camera_gradient: array<f32>;
@group(0) @binding(5) var<storage, read_write> point_gradient: array<f32>;
@group(0) @binding(6) var<storage, read_write> camera_hessian: array<f32>;
@group(0) @binding(7) var<storage, read_write> point_hessian: array<f32>;
@group(0) @binding(8) var<storage, read_write> cross: array<f32>;

fn quaternion_rotation(q: vec4<f32>) -> array<array<f32, 3>, 3> {
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    return array<array<f32, 3>, 3>(
        array<f32, 3>(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y)
        ),
        array<f32, 3>(
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x)
        ),
        array<f32, 3>(
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y)
        )
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let point = gid.x;
    if (point >= ba_meta.point_count) {
        return;
    }

    let camera_offset = ba_meta.camera_count * 7u;
    let point_start = camera_offset + point * 3u;
    let world_point = vec3<f32>(
        ambient[point_start],
        ambient[point_start + 1u],
        ambient[point_start + 2u]
    );
    var point_cost = 0.0;

    for (var i = 0u; i < 3u; i++) {
        point_gradient[point * 3u + i] = 0.0;
        for (var j = 0u; j < 3u; j++) {
            point_hessian[point * 9u + i * 3u + j] = 0.0;
        }
    }

    let start_count = ba_meta.camera_count - ba_meta.window + 1u;
    let first_camera = point % start_count;
    for (var slot = 0u; slot < ba_meta.window; slot++) {
        let camera_index = first_camera + slot;
        let camera_start = camera_index * 7u;
        let q = vec4<f32>(
            ambient[camera_start],
            ambient[camera_start + 1u],
            ambient[camera_start + 2u],
            ambient[camera_start + 3u]
        );
        let translation = vec3<f32>(
            ambient[camera_start + 4u],
            ambient[camera_start + 5u],
            ambient[camera_start + 6u]
        );
        let r = quaternion_rotation(q);
        let x = r[0][0] * world_point.x + r[0][1] * world_point.y
            + r[0][2] * world_point.z + translation.x;
        let y = r[1][0] * world_point.x + r[1][1] * world_point.y
            + r[1][2] * world_point.z + translation.y;
        let z = r[2][0] * world_point.x + r[2][1] * world_point.y
            + r[2][2] * world_point.z + translation.z;
        let z_inv = 1.0 / z;
        let z_inv2 = z_inv * z_inv;
        let projection = array<array<f32, 3>, 2>(
            array<f32, 3>(520.0 * z_inv, 0.0, -520.0 * x * z_inv2),
            array<f32, 3>(0.0, 520.0 * z_inv, -520.0 * y * z_inv2)
        );

        let observation = point * ba_meta.window + slot;
        let measured = vec2<f32>(
            observations[observation * 2u],
            observations[observation * 2u + 1u]
        );
        let residual = vec2<f32>(
            520.0 * x * z_inv + 320.0,
            520.0 * y * z_inv + 240.0
        ) - measured;
        point_cost += residual.x * residual.x + residual.y * residual.y;

        let px = world_point.x;
        let py = world_point.y;
        let pz = world_point.z;
        var neg_r_skew = array<array<f32, 3>, 3>();
        neg_r_skew[0] = array<f32, 3>(
            -r[0][1] * pz + r[0][2] * py,
            r[0][0] * pz - r[0][2] * px,
            -r[0][0] * py + r[0][1] * px
        );
        neg_r_skew[1] = array<f32, 3>(
            -r[1][1] * pz + r[1][2] * py,
            r[1][0] * pz - r[1][2] * px,
            -r[1][0] * py + r[1][1] * px
        );
        neg_r_skew[2] = array<f32, 3>(
            -r[2][1] * pz + r[2][2] * py,
            r[2][0] * pz - r[2][2] * px,
            -r[2][0] * py + r[2][1] * px
        );

        var jc = array<f32, 12>();
        var jp = array<f32, 6>();
        for (var row = 0u; row < 2u; row++) {
            for (var col = 0u; col < 3u; col++) {
                jc[row * 6u + col] =
                    projection[row][0] * neg_r_skew[0][col] +
                    projection[row][1] * neg_r_skew[1][col] +
                    projection[row][2] * neg_r_skew[2][col];
                jc[row * 6u + 3u + col] =
                    projection[row][0] * r[0][col] +
                    projection[row][1] * r[1][col] +
                    projection[row][2] * r[2][col];
                jp[row * 3u + col] =
                    projection[row][0] * r[0][col] +
                    projection[row][1] * r[1][col] +
                    projection[row][2] * r[2][col];
            }
        }

        let base_g = observation * 6u;
        let base_h = observation * 36u;
        let base_c = observation * 18u;
        for (var i = 0u; i < 6u; i++) {
            camera_gradient[base_g + i] =
                jc[i] * residual.x + jc[6u + i] * residual.y;
            for (var j = 0u; j < 6u; j++) {
                camera_hessian[base_h + i * 6u + j] =
                    jc[i] * jc[j] + jc[6u + i] * jc[6u + j];
            }
            for (var j = 0u; j < 3u; j++) {
                cross[base_c + i * 3u + j] =
                    jc[i] * jp[j] + jc[6u + i] * jp[3u + j];
            }
        }
        for (var i = 0u; i < 3u; i++) {
            point_gradient[point * 3u + i] +=
                jp[i] * residual.x + jp[3u + i] * residual.y;
            for (var j = 0u; j < 3u; j++) {
                point_hessian[point * 9u + i * 3u + j] +=
                    jp[i] * jp[j] + jp[3u + i] * jp[3u + j];
            }
        }
    }
    cost[point] = point_cost;
}
