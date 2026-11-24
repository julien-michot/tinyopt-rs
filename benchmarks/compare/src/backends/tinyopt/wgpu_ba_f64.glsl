#version 450
#extension GL_ARB_gpu_shader_fp64 : enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer Params {
    double values[];
} params;

layout(set = 0, binding = 1, std430) readonly buffer Observations {
    double values[];
} observations;

layout(set = 0, binding = 2, std430) readonly buffer Meta {
    uint camera_count;
    uint point_count;
    uint window;
    uint dims;
} meta;

layout(set = 0, binding = 3, std430) buffer Cost {
    double values[];
} cost;
layout(set = 0, binding = 4, std430) buffer CameraGradient {
    double values[];
} camera_gradient;
layout(set = 0, binding = 5, std430) buffer PointGradient {
    double values[];
} point_gradient;
layout(set = 0, binding = 6, std430) buffer CameraHessian {
    double values[];
} camera_hessian;
layout(set = 0, binding = 7, std430) buffer PointHessian {
    double values[];
} point_hessian;
layout(set = 0, binding = 8, std430) buffer Cross {
    double values[];
} cross_blocks;

dvec3 rotate_point(dvec4 q, dvec3 point) {
    dvec3 vector_part = q.yzw;
    double dot_value = dot(vector_part, point);
    return 2.0 * dot_value * vector_part
        + (q.x * q.x - dot(vector_part, vector_part)) * point
        + 2.0 * q.x * cross(vector_part, point);
}

void main() {
    uint point_index = gl_GlobalInvocationID.x;
    if (point_index >= meta.point_count) {
        return;
    }

    uint camera_offset = meta.camera_count * 7u;
    uint point_start = camera_offset + point_index * 3u;
    dvec3 world_point = dvec3(
        params.values[point_start],
        params.values[point_start + 1u],
        params.values[point_start + 2u]
    );
    for (uint i = 0u; i < 3u; i++) {
        point_gradient.values[point_index * 3u + i] = 0.0;
        for (uint j = 0u; j < 3u; j++) {
            point_hessian.values[point_index * 9u + i * 3u + j] = 0.0;
        }
    }

    double point_cost = 0.0;
    uint start_count = meta.camera_count - meta.window + 1u;
    uint first_camera = point_index % start_count;
    for (uint slot = 0u; slot < meta.window; slot++) {
        uint camera_index = first_camera + slot;
        uint camera_start = camera_index * 7u;
        dvec4 q = dvec4(
            params.values[camera_start],
            params.values[camera_start + 1u],
            params.values[camera_start + 2u],
            params.values[camera_start + 3u]
        );
        dvec3 translation = dvec3(
            params.values[camera_start + 4u],
            params.values[camera_start + 5u],
            params.values[camera_start + 6u]
        );
        dvec3 camera_point = rotate_point(q, world_point) + translation;
        double x = camera_point.x;
        double y = camera_point.y;
        double z = camera_point.z;
        dvec3 projection_x = dvec3(520.0 / z, 0.0, -520.0 * x / (z * z));
        dvec3 projection_y = dvec3(0.0, 520.0 / z, -520.0 * y / (z * z));

        uint observation = point_index * meta.window + slot;
        dvec2 measured = dvec2(
            observations.values[observation * 2u],
            observations.values[observation * 2u + 1u]
        );
        dvec2 residual = dvec2(
            520.0 * x / z + 320.0,
            520.0 * y / z + 240.0
        ) - measured;
        point_cost += dot(residual, residual);

        dvec3 camera_rotation_columns[3];
        camera_rotation_columns[0] = rotate_point(q, cross(dvec3(1.0, 0.0, 0.0), world_point));
        camera_rotation_columns[1] = rotate_point(q, cross(dvec3(0.0, 1.0, 0.0), world_point));
        camera_rotation_columns[2] = rotate_point(q, cross(dvec3(0.0, 0.0, 1.0), world_point));
        dvec3 point_rotation_columns[3];
        point_rotation_columns[0] = rotate_point(q, dvec3(1.0, 0.0, 0.0));
        point_rotation_columns[1] = rotate_point(q, dvec3(0.0, 1.0, 0.0));
        point_rotation_columns[2] = rotate_point(q, dvec3(0.0, 0.0, 1.0));

        double jc[12];
        double jp[6];
        for (uint row = 0u; row < 2u; row++) {
            dvec3 projection = row == 0u ? projection_x : projection_y;
            for (uint col = 0u; col < 3u; col++) {
                jc[row * 6u + col] = dot(projection, camera_rotation_columns[col]);
                jc[row * 6u + 3u + col] = dot(projection, point_rotation_columns[col]);
                jp[row * 3u + col] = dot(projection, point_rotation_columns[col]);
            }
        }

        uint base_gradient = observation * 6u;
        uint base_hessian = observation * 36u;
        uint base_cross = observation * 18u;
        for (uint i = 0u; i < 6u; i++) {
            camera_gradient.values[base_gradient + i] =
                jc[i] * residual.x + jc[6u + i] * residual.y;
            for (uint j = 0u; j < 6u; j++) {
                camera_hessian.values[base_hessian + i * 6u + j] =
                    jc[i] * jc[j] + jc[6u + i] * jc[6u + j];
            }
            for (uint j = 0u; j < 3u; j++) {
                cross_blocks.values[base_cross + i * 3u + j] =
                    jc[i] * jp[j] + jc[6u + i] * jp[3u + j];
            }
        }
        for (uint i = 0u; i < 3u; i++) {
            point_gradient.values[point_index * 3u + i] +=
                jp[i] * residual.x + jp[3u + i] * residual.y;
            for (uint j = 0u; j < 3u; j++) {
                point_hessian.values[point_index * 9u + i * 3u + j] +=
                    jp[i] * jp[j] + jp[3u + i] * jp[3u + j];
            }
        }
    }
    cost.values[point_index] = point_cost;
}
