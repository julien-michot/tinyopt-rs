#![allow(clippy::items_after_test_module)]

use std::cell::RefCell;
use std::sync::mpsc;

use tinyopt::backend::{DenseVector, Vector};
use tinyopt::block_sparse::BlockSparseMat;
use tinyopt::optimizers::lm;
use tinyopt::{Cost, Dynamic};

#[cfg(test)]
use super::run_ba_block;
use super::{BaParams, BaTypes, ba_params, outcome};

use crate::harness::Outcome;
use crate::options::tinyopt_options;
#[cfg(test)]
use crate::problems::Kind;
use crate::problems::{Case, bundle_adjustment as ba};

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Meta {
    camera_count: u32,
    point_count: u32,
    window: u32,
    dims: u32,
}

pub struct WgpuBaF32 {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    cost: wgpu::Buffer,
    camera_gradient: wgpu::Buffer,
    point_gradient: wgpu::Buffer,
    camera_hessian: wgpu::Buffer,
    point_hessian: wgpu::Buffer,
    cross: wgpu::Buffer,
    point_count: u32,
    window: u32,
    staging: Vec<wgpu::Buffer>,
    params_values: Vec<f32>,
}

impl WgpuBaF32 {
    pub fn new(cams: usize, points: usize) -> Self {
        let problem = ba::make_problem(cams, points);
        let window = cams.min(ba::POINT_CAMERA_WINDOW) as u32;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no suitable wgpu adapter");
        let mut limits = adapter.limits();
        limits.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.max(9);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tinyopt BA f32"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }))
        .expect("create wgpu device");

        let params_len = (7 * cams + 3 * points).next_multiple_of(4);
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BA parameters"),
            size: (params_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let observation_values: Vec<f32> = problem
            .observations
            .iter()
            .flat_map(|observation| observation.measurement)
            .map(|value| value as f32)
            .collect();
        let observations = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BA observations"),
            contents: bytemuck::cast_slice(&observation_values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let meta = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("BA meta"),
            contents: bytemuck::bytes_of(&Meta {
                camera_count: cams as u32,
                point_count: points as u32,
                window,
                dims: (7 * cams + 3 * points) as u32,
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let make_output = |label: &str, length: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (length.next_multiple_of(4) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let cost = make_output("BA cost", points);
        let camera_gradient = make_output("BA camera gradient", points * window as usize * 6);
        let point_gradient = make_output("BA point gradient", points * 3);
        let camera_hessian = make_output("BA camera Hessian", points * window as usize * 36);
        let point_hessian = make_output("BA point Hessian", points * 9);
        let cross = make_output("BA cross Hessian", points * window as usize * 18);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BA bind group layout"),
            entries: &[
                storage_layout(0),
                storage_layout(1),
                storage_layout(2),
                storage_layout(3),
                storage_layout(4),
                storage_layout(5),
                storage_layout(6),
                storage_layout(7),
                storage_layout(8),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BA pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BA WGSL"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgpu_ba.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BA pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BA bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: observations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cost.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: camera_gradient.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: point_gradient.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: camera_hessian.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: point_hessian.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: cross.as_entire_binding(),
                },
            ],
        });
        let lengths = [
            points,
            points * window as usize * 6,
            points * 3,
            points * window as usize * 36,
            points * 9,
            points * window as usize * 18,
        ];
        let staging = lengths
            .iter()
            .map(|&length| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("BA persistent staging"),
                    size: (length.next_multiple_of(4) * 4) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Self {
            device,
            queue,
            pipeline,
            bind_group,
            params,
            cost,
            camera_gradient,
            point_gradient,
            camera_hessian,
            point_hessian,
            cross,
            point_count: points as u32,
            window,
            staging,
            params_values: Vec::with_capacity(7 * cams + 3 * points),
        }
    }

    pub fn accumulate(&mut self, x: &[f64], need_grad_hessian: bool) -> WgpuBaContribution {
        self.params_values.clear();
        self.params_values
            .extend(x.iter().map(|&value| value as f32));
        self.queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&self.params_values));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BA assembly"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BA assembly pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.point_count.div_ceil(64), 1, 1);
        }
        let output_buffers = if need_grad_hessian {
            [
                &self.cost,
                &self.camera_gradient,
                &self.point_gradient,
                &self.camera_hessian,
                &self.point_hessian,
                &self.cross,
            ]
        } else {
            [
                &self.cost, &self.cost, &self.cost, &self.cost, &self.cost, &self.cost,
            ]
        };
        let lengths = if need_grad_hessian {
            [
                self.point_count as usize,
                (self.point_count * self.window * 6) as usize,
                (self.point_count * 3) as usize,
                (self.point_count * self.window * 36) as usize,
                (self.point_count * 9) as usize,
                (self.point_count * self.window * 18) as usize,
            ]
        } else {
            [self.point_count as usize, 0, 0, 0, 0, 0]
        };
        let staging_count = if need_grad_hessian { 6 } else { 1 };
        if need_grad_hessian {
            for (staging, (&length, output)) in self
                .staging
                .iter()
                .zip(lengths.iter().zip(output_buffers.iter()))
            {
                encoder.copy_buffer_to_buffer(
                    output,
                    0,
                    staging,
                    0,
                    (length.next_multiple_of(4) * 4) as u64,
                );
            }
        } else {
            encoder.copy_buffer_to_buffer(
                &self.cost,
                0,
                &self.staging[0],
                0,
                (lengths[0].next_multiple_of(4) * 4) as u64,
            );
        }
        self.queue.submit([encoder.finish()]);

        let mut receivers = Vec::with_capacity(staging_count);
        for staging in self.staging.iter().take(staging_count) {
            let (sender, receiver) = mpsc::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });
            receivers.push(receiver);
        }
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("wgpu poll");

        let mut mapped_values = Vec::with_capacity(staging_count);
        for (index, (receiver, staging)) in receivers
            .iter()
            .zip(self.staging.iter().take(staging_count))
            .enumerate()
        {
            let length = lengths[index];
            receiver
                .recv()
                .expect("wgpu readback")
                .expect("map BA staging buffer");
            let view = staging
                .slice(..)
                .get_mapped_range()
                .expect("map buffer view");
            mapped_values.push(bytemuck::cast_slice::<u8, f32>(&view[..length * 4]).to_vec());
            drop(view);
            staging.unmap();
        }
        let cost = mapped_values.remove(0);
        if need_grad_hessian {
            WgpuBaContribution {
                cost,
                camera_gradient: mapped_values.remove(0),
                point_gradient: mapped_values.remove(0),
                camera_hessian: mapped_values.remove(0),
                point_hessian: mapped_values.remove(0),
                cross: mapped_values.remove(0),
            }
        } else {
            WgpuBaContribution {
                cost,
                camera_gradient: Vec::new(),
                point_gradient: Vec::new(),
                camera_hessian: Vec::new(),
                point_hessian: Vec::new(),
                cross: Vec::new(),
            }
        }
    }
}

fn storage_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage {
                read_only: binding < 3,
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub struct WgpuBaContribution {
    pub cost: Vec<f32>,
    pub camera_gradient: Vec<f32>,
    pub point_gradient: Vec<f32>,
    pub camera_hessian: Vec<f32>,
    pub point_hessian: Vec<f32>,
    pub cross: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CpuContribution {
        gradient: Vec<f64>,
        camera_gradient: Vec<f64>,
        camera_hessian: Vec<f64>,
        point_hessian: Vec<f64>,
        cross: Vec<f64>,
    }

    fn cpu_initial_gradient_hessian(problem: &ba::Problem) -> CpuContribution {
        let cams = problem.camera_count();
        let points = problem.point_count();
        let camera_offset = 6 * cams;
        let window = cams.min(ba::POINT_CAMERA_WINDOW);
        let mut gradient = vec![0.0; camera_offset + 3 * points];
        let mut camera_gradient = vec![0.0; points * window * 6];
        let mut camera_hessian = vec![0.0; points * window * 36];
        let mut point_hessian = vec![0.0; points * 9];
        let mut cross = vec![0.0; points * window * 18];

        for observation in &problem.observations {
            let camera = problem.initial_cameras[observation.camera];
            let point = problem.initial_points[observation.point];
            let (jc, jp) = ba::projection_jacobians(&camera, &point);
            let projected = ba::project(&camera, &point);
            let residual = [
                projected[0] - observation.measurement[0],
                projected[1] - observation.measurement[1],
            ];

            let point_start = camera_offset + observation.point * 3;
            for i in 0..3 {
                gradient[point_start + i] += jp[i] * residual[0] + jp[3 + i] * residual[1];
            }

            let first_camera = observation.point % (cams - window + 1);
            let slot = observation.camera - first_camera;
            let base_g = (observation.point * window + slot) * 6;
            for i in 0..6 {
                camera_gradient[base_g + i] = jc[i] * residual[0] + jc[6 + i] * residual[1];
            }
            let base_h = (observation.point * window + slot) * 36;
            let base_c = (observation.point * window + slot) * 18;
            for i in 0..6 {
                for j in 0..6 {
                    camera_hessian[base_h + i * 6 + j] += jc[i] * jc[j] + jc[6 + i] * jc[6 + j];
                }
                for j in 0..3 {
                    cross[base_c + i * 3 + j] += jc[i] * jp[j] + jc[6 + i] * jp[3 + j];
                }
            }
            let base_p = observation.point * 9;
            for i in 0..3 {
                for j in 0..3 {
                    point_hessian[base_p + i * 3 + j] += jp[i] * jp[j] + jp[3 + i] * jp[3 + j];
                }
            }
        }
        CpuContribution {
            gradient,
            camera_gradient,
            camera_hessian,
            point_hessian,
            cross,
        }
    }

    fn assert_contribution_close(
        problem: &ba::Problem,
        gpu: &WgpuBaContribution,
        cpu: &CpuContribution,
    ) {
        let cams = problem.camera_count();
        let points = problem.point_count();
        let camera_offset = 6 * cams;
        let window = cams.min(ba::POINT_CAMERA_WINDOW);
        let mut max_gradient_error = 0.0_f64;
        let mut max_camera_hessian_error = 0.0_f64;
        let mut max_point_hessian_error = 0.0_f64;
        let mut max_cross_error = 0.0_f64;
        let mut camera_hessian_scale = 1.0_f64;
        let mut point_hessian_scale = 1.0_f64;
        let mut cross_scale = 1.0_f64;
        let mut gradient_scale = 1.0_f64;

        for point in 0..points {
            for i in 0..3 {
                let index = camera_offset + point * 3 + i;
                gradient_scale = gradient_scale.max(cpu.gradient[index].abs());
                max_gradient_error = max_gradient_error
                    .max((gpu.point_gradient[point * 3 + i] as f64 - cpu.gradient[index]).abs());
            }
            for slot in 0..window {
                for i in 0..6 {
                    let cpu_value = cpu.camera_gradient[(point * window + slot) * 6 + i];
                    gradient_scale = gradient_scale.max(cpu_value.abs());
                    max_gradient_error = max_gradient_error.max(
                        (gpu.camera_gradient[(point * window + slot) * 6 + i] as f64 - cpu_value)
                            .abs(),
                    );
                }
                for i in 0..36 {
                    let cpu_value = cpu.camera_hessian[(point * window + slot) * 36 + i];
                    camera_hessian_scale = camera_hessian_scale.max(cpu_value.abs());
                    max_camera_hessian_error = max_camera_hessian_error.max(
                        (gpu.camera_hessian[(point * window + slot) * 36 + i] as f64 - cpu_value)
                            .abs(),
                    );
                }
                for i in 0..18 {
                    let cpu_value = cpu.cross[(point * window + slot) * 18 + i];
                    cross_scale = cross_scale.max(cpu_value.abs());
                    max_cross_error = max_cross_error.max(
                        (gpu.cross[(point * window + slot) * 18 + i] as f64 - cpu_value).abs(),
                    );
                }
            }
            for i in 0..9 {
                let cpu_value = cpu.point_hessian[point * 9 + i];
                point_hessian_scale = point_hessian_scale.max(cpu_value.abs());
                max_point_hessian_error = max_point_hessian_error.max(
                    (gpu.point_hessian[point * 9 + i] as f64 - cpu.point_hessian[point * 9 + i])
                        .abs(),
                );
            }
        }
        assert!(
            max_gradient_error < 1e-3 * gradient_scale,
            "gpu/cpu gradient max error: {max_gradient_error}"
        );
        assert!(
            max_camera_hessian_error < 1e-3 * camera_hessian_scale,
            "gpu/cpu camera Hessian max error: {max_camera_hessian_error}"
        );
        assert!(
            max_point_hessian_error < 1e-3 * point_hessian_scale,
            "gpu/cpu point Hessian max error: {max_point_hessian_error}"
        );
        assert!(
            max_cross_error < 1e-3 * cross_scale,
            "gpu/cpu cross Hessian max error: {max_cross_error}"
        );
    }

    #[test]
    fn wgpu_f32_initial_cost_matches_cpu() {
        let cams = 4;
        let points = 600;
        let problem = ba::make_problem(cams, points);
        let initial = ba_params(&problem);
        let mut gpu = WgpuBaF32::new(cams, points);
        let contribution = gpu.accumulate(&initial.values, false);
        let gpu_cost = 0.5
            * contribution
                .cost
                .iter()
                .map(|&value| value as f64)
                .sum::<f64>();
        let cpu_cost = ba::initial_cost(&problem);
        assert!(
            (gpu_cost - cpu_cost).abs() < 1e-3 * cpu_cost,
            "gpu cost {gpu_cost}, cpu cost {cpu_cost}"
        );

        let cpu_gradient_and_hessian = cpu_initial_gradient_hessian(&problem);
        let gpu = gpu.accumulate(&initial.values, true);
        let cpu = cpu_gradient_and_hessian;
        assert_contribution_close(&problem, &gpu, &cpu);
    }

    #[test]
    fn wgpu_f32_matches_serial_accumulation() {
        let cams = 4;
        let points = 600;
        let case = Case {
            kind: Kind::BundleAdjustmentBlock { cams, points },
            name: format!("{cams}c {points}p"),
            short: format!("{cams}c {points}p"),
        };
        let serial = run_ba_block(&case, cams, points);
        let wgpu = run_ba_block_wgpu(&case, cams, points);
        assert!(serial.converged && wgpu.converged);
        assert!(
            (serial.final_cost - wgpu.final_cost).abs() < 1e-2 * serial.final_cost.abs().max(1.0),
            "serial final cost {}, wgpu final cost {}",
            serial.final_cost,
            wgpu.final_cost
        );
    }

    #[test]
    fn wgpu_f64_initial_assembly_matches_cpu() {
        if !wgpu_f64_supported() {
            eprintln!("skipping native f64 assembly: no adapter exposes SHADER_F64");
            return;
        }

        let cams = 4;
        let points = 600;
        let problem = ba::make_problem(cams, points);
        let initial = ba_params(&problem);
        let mut gpu = WgpuBaF64::new(cams, points).expect("create f64 BA pipeline");
        let contribution = gpu.accumulate(&initial.values, true);
        let cpu = cpu_initial_gradient_hessian(&problem);
        let gpu_cost = 0.5 * contribution.cost.iter().sum::<f64>();
        let cpu_cost = ba::initial_cost(&problem);
        assert!(
            (gpu_cost - cpu_cost).abs() < 1e-10 * cpu_cost,
            "gpu cost {gpu_cost}, cpu cost {cpu_cost}"
        );

        let assert_close = |name: &str, gpu: &[f64], cpu: &[f64]| {
            assert_eq!(gpu.len(), cpu.len(), "{name} length");
            let mut max_error = 0.0_f64;
            let mut max_index = 0;
            let mut scale = 1.0_f64;
            for (index, (&gpu_value, &cpu_value)) in gpu.iter().zip(cpu).enumerate() {
                let error = (gpu_value - cpu_value).abs();
                if error > max_error {
                    max_error = error;
                    max_index = index;
                }
                scale = scale.max(cpu_value.abs());
            }
            assert!(
                max_error < 1e-10 * scale,
                "{name} max error {max_error} at {max_index}: gpu {}, cpu {}, scale {scale}",
                gpu[max_index],
                cpu[max_index]
            );
        };
        assert_close(
            "camera gradient",
            &contribution.camera_gradient,
            &cpu.camera_gradient,
        );
        assert_close(
            "point gradient",
            &contribution.point_gradient,
            &cpu.gradient[6 * cams..],
        );
        assert_close(
            "camera Hessian",
            &contribution.camera_hessian,
            &cpu.camera_hessian,
        );
        assert_close(
            "point Hessian",
            &contribution.point_hessian,
            &cpu.point_hessian,
        );
        assert_close("cross Hessian", &contribution.cross, &cpu.cross);
    }
}

pub(crate) fn measure_ba_block_wgpu(
    cams: usize,
    points: usize,
    warmup: std::time::Duration,
    samples: usize,
) -> crate::harness::Stats {
    let problem = ba::make_problem(cams, points);
    let params = ba_params(&problem);
    let mut gpu = WgpuBaF32::new(cams, points);
    crate::harness::measure(warmup, samples, || {
        let contribution = gpu.accumulate(&params.values, true);
        assert!(!contribution.cost.is_empty());
        Outcome {
            converged: true,
            iterations: 1,
            final_cost: 0.0,
        }
    })
    .0
}

pub(crate) fn measure_ba_block_wgpu_f64(
    cams: usize,
    points: usize,
    warmup: std::time::Duration,
    samples: usize,
) -> crate::harness::Stats {
    let problem = ba::make_problem(cams, points);
    let params = ba_params(&problem);
    let mut gpu = WgpuBaF64::new(cams, points).expect("create f64 WGPU BA pipeline");
    crate::harness::measure(warmup, samples, || {
        let contribution = gpu.accumulate(&params.values, true);
        assert!(!contribution.cost.is_empty());
        Outcome {
            converged: true,
            iterations: 1,
            final_cost: 0.0,
        }
    })
    .0
}

pub(crate) fn run_ba_block_wgpu(case: &Case, cams: usize, points: usize) -> Outcome {
    type V = Vector<f64, Dynamic>;
    let problem = ba::make_problem(cams, points);
    let camera_offset = 6 * cams;
    let window = cams.min(ba::POINT_CAMERA_WINDOW);
    let gpu = RefCell::new(WgpuBaF32::new(cams, points));

    let mut opt =
        lm::Optimizer::<BaParams, BlockSparseMat<BaTypes, f64>>::new(tinyopt_options(&case.kind));
    opt.init_with(BlockSparseMat::with_counts(&[cams, points]));
    let mut x = ba_params(&problem);
    let initial_contribution = gpu.borrow_mut().accumulate(&x.values, true);
    let gpu_initial_cost = 0.5
        * initial_contribution
            .cost
            .iter()
            .map(|&value| value as f64)
            .sum::<f64>();
    let cpu_initial_cost = ba::initial_cost(&problem);
    assert!(
        (gpu_initial_cost - cpu_initial_cost).abs() < 2e-1 * cpu_initial_cost,
        "wgpu initial cost {gpu_initial_cost}, CPU initial cost {cpu_initial_cost}"
    );

    let acc = |xv: &[f64], grad: Option<&mut V>, h: Option<&mut BlockSparseMat<BaTypes, f64>>| {
        let contribution = gpu
            .borrow_mut()
            .accumulate(xv, grad.is_some() || h.is_some());
        let cost = contribution
            .cost
            .iter()
            .map(|&value| value as f64)
            .sum::<f64>();

        if let Some(g) = grad {
            for point in 0..points {
                let point_start = camera_offset + point * 3;
                for i in 0..3 {
                    g.set_at(
                        point_start + i,
                        g.at(point_start + i) + contribution.point_gradient[point * 3 + i] as f64,
                    );
                }
                let first_camera = point % (cams - window + 1);
                for slot in 0..window {
                    let camera = first_camera + slot;
                    for i in 0..6 {
                        let index = camera * 6 + i;
                        g.set_at(
                            index,
                            g.at(index)
                                + contribution.camera_gradient[(point * window + slot) * 6 + i]
                                    as f64,
                        );
                    }
                }
            }
        }
        if let Some(h) = h {
            for point in 0..points {
                let mut point_block = h.block_mut::<3, 3>(point, point);
                for i in 0..3 {
                    for j in 0..3 {
                        point_block.set_at(
                            i,
                            j,
                            point_block.at(i, j)
                                + contribution.point_hessian[point * 9 + i * 3 + j] as f64,
                        );
                    }
                }
                let first_camera = point % (cams - window + 1);
                for slot in 0..window {
                    let camera = first_camera + slot;
                    let mut camera_block = h.block_mut::<6, 6>(camera, camera);
                    for i in 0..6 {
                        for j in 0..6 {
                            camera_block.set_at(
                                i,
                                j,
                                camera_block.at(i, j)
                                    + contribution.camera_hessian
                                        [(point * window + slot) * 36 + i * 6 + j]
                                        as f64,
                            );
                        }
                    }
                    let mut cross_block = h.block_mut::<6, 3>(camera, point);
                    for i in 0..6 {
                        for j in 0..3 {
                            cross_block.set_at(
                                i,
                                j,
                                contribution.cross[(point * window + slot) * 18 + i * 3 + j] as f64,
                            );
                        }
                    }
                }
            }
        }
        Cost::with_num_residuals(0.5 * cost, (2 * problem.observations.len()) as i32)
    };

    let summary = opt.optimize(&mut x, acc);
    outcome(&summary)
}

pub(crate) fn wgpu_f64_supported() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .is_ok_and(|adapter| adapter.features().contains(wgpu::Features::SHADER_F64))
}

pub struct WgpuBaF64 {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    cost: wgpu::Buffer,
    camera_gradient: wgpu::Buffer,
    point_gradient: wgpu::Buffer,
    camera_hessian: wgpu::Buffer,
    point_hessian: wgpu::Buffer,
    cross: wgpu::Buffer,
    point_count: u32,
    window: u32,
    staging: Vec<wgpu::Buffer>,
}

impl WgpuBaF64 {
    pub fn new(cams: usize, points: usize) -> Result<Self, String> {
        let problem = ba::make_problem(cams, points);
        let window = cams.min(ba::POINT_CAMERA_WINDOW) as u32;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })) {
                Ok(adapter) => adapter,
                Err(error) => return Err(format!("no suitable wgpu adapter: {error}")),
            };
        if !adapter.features().contains(wgpu::Features::SHADER_F64) {
            return Err(format!(
                "adapter {:?} does not expose wgpu::Features::SHADER_F64",
                adapter.get_info()
            ));
        }
        let mut limits = adapter.limits();
        limits.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.max(9);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tinyopt BA f64"),
            required_features: wgpu::Features::SHADER_F64,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|error| format!("create f64 wgpu device: {error}"))?;

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BA f64 parameters"),
            size: ((7 * cams + 3 * points) * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let observation_values: Vec<f64> = problem
            .observations
            .iter()
            .flat_map(|observation| observation.measurement)
            .collect();
        let observations = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("BA f64 observations"),
            contents: bytemuck::cast_slice(&observation_values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let meta = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("BA f64 meta"),
            contents: bytemuck::bytes_of(&Meta {
                camera_count: cams as u32,
                point_count: points as u32,
                window,
                dims: (7 * cams + 3 * points) as u32,
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let make_output = |label: &str, length: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (length * 8) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let cost = make_output("BA f64 cost", points);
        let camera_gradient = make_output("BA f64 camera gradient", points * window as usize * 6);
        let point_gradient = make_output("BA f64 point gradient", points * 3);
        let camera_hessian = make_output("BA f64 camera Hessian", points * window as usize * 36);
        let point_hessian = make_output("BA f64 point Hessian", points * 9);
        let cross = make_output("BA f64 cross Hessian", points * window as usize * 18);

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BA f64 bind layout"),
            entries: &(0..9).map(storage_layout).collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BA f64 pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BA f64 GLSL"),
            source: wgpu::ShaderSource::Glsl {
                shader: include_str!("wgpu_ba_f64.glsl").into(),
                stage: wgpu::naga::ShaderStage::Compute,
                defines: &[],
            },
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BA f64 pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BA f64 bind group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: observations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cost.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: camera_gradient.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: point_gradient.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: camera_hessian.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: point_hessian.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: cross.as_entire_binding(),
                },
            ],
        });
        let lengths = [
            points,
            points * window as usize * 6,
            points * 3,
            points * window as usize * 36,
            points * 9,
            points * window as usize * 18,
        ];
        let staging = lengths
            .iter()
            .map(|&length| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("BA f64 persistent staging"),
                    size: (length * 8) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
            params,
            cost,
            camera_gradient,
            point_gradient,
            camera_hessian,
            point_hessian,
            cross,
            point_count: points as u32,
            window,
            staging,
        })
    }

    pub fn accumulate(&mut self, x: &[f64], need_grad_hessian: bool) -> WgpuBaContributionF64 {
        self.queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(x));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BA f64 assembly"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BA f64 assembly pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.point_count.div_ceil(64), 1, 1);
        }
        let lengths = if need_grad_hessian {
            [
                self.point_count as usize,
                (self.point_count * self.window * 6) as usize,
                (self.point_count * 3) as usize,
                (self.point_count * self.window * 36) as usize,
                (self.point_count * 9) as usize,
                (self.point_count * self.window * 18) as usize,
            ]
        } else {
            [self.point_count as usize, 0, 0, 0, 0, 0]
        };
        let outputs = [
            &self.cost,
            &self.camera_gradient,
            &self.point_gradient,
            &self.camera_hessian,
            &self.point_hessian,
            &self.cross,
        ];
        let staging_count = if need_grad_hessian { 6 } else { 1 };
        for (index, (&length, staging)) in lengths
            .iter()
            .take(staging_count)
            .zip(self.staging.iter().take(staging_count))
            .enumerate()
        {
            encoder.copy_buffer_to_buffer(outputs[index], 0, staging, 0, (length * 8) as u64);
        }
        self.queue.submit([encoder.finish()]);
        let mut receivers = Vec::with_capacity(staging_count);
        for staging in self.staging.iter().take(staging_count) {
            let (sender, receiver) = mpsc::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });
            receivers.push(receiver);
        }
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("wgpu f64 poll");
        let mut mapped_values = Vec::with_capacity(staging_count);
        for (index, (receiver, staging)) in receivers
            .iter()
            .zip(self.staging.iter().take(staging_count))
            .enumerate()
        {
            receiver
                .recv()
                .expect("wgpu f64 readback")
                .expect("map f64 staging");
            let view = staging.slice(..).get_mapped_range().expect("map f64 view");
            let length = lengths[index];
            mapped_values.push(bytemuck::cast_slice::<u8, f64>(&view[..length * 8]).to_vec());
            drop(view);
            staging.unmap();
        }
        let cost = mapped_values.remove(0);
        if need_grad_hessian {
            WgpuBaContributionF64 {
                cost,
                camera_gradient: mapped_values.remove(0),
                point_gradient: mapped_values.remove(0),
                camera_hessian: mapped_values.remove(0),
                point_hessian: mapped_values.remove(0),
                cross: mapped_values.remove(0),
            }
        } else {
            WgpuBaContributionF64 {
                cost,
                camera_gradient: Vec::new(),
                point_gradient: Vec::new(),
                camera_hessian: Vec::new(),
                point_hessian: Vec::new(),
                cross: Vec::new(),
            }
        }
    }
}

pub struct WgpuBaContributionF64 {
    pub cost: Vec<f64>,
    pub camera_gradient: Vec<f64>,
    pub point_gradient: Vec<f64>,
    pub camera_hessian: Vec<f64>,
    pub point_hessian: Vec<f64>,
    pub cross: Vec<f64>,
}

pub(crate) fn run_ba_block_wgpu_f64(case: &Case, cams: usize, points: usize) -> Outcome {
    type V = Vector<f64, Dynamic>;
    let problem = ba::make_problem(cams, points);
    let camera_offset = 6 * cams;
    let window = cams.min(ba::POINT_CAMERA_WINDOW);
    let gpu = match WgpuBaF64::new(cams, points) {
        Ok(gpu) => RefCell::new(gpu),
        Err(error) => panic!("{error}"),
    };
    let mut opt =
        lm::Optimizer::<BaParams, BlockSparseMat<BaTypes, f64>>::new(tinyopt_options(&case.kind));
    opt.init_with(BlockSparseMat::with_counts(&[cams, points]));
    let mut x = ba_params(&problem);
    let acc = |xv: &[f64], grad: Option<&mut V>, h: Option<&mut BlockSparseMat<BaTypes, f64>>| {
        let contribution = gpu
            .borrow_mut()
            .accumulate(xv, grad.is_some() || h.is_some());
        let cost = contribution.cost.iter().copied().sum::<f64>();
        if let Some(g) = grad {
            for point in 0..points {
                let point_start = camera_offset + point * 3;
                for i in 0..3 {
                    g.set_at(
                        point_start + i,
                        g.at(point_start + i) + contribution.point_gradient[point * 3 + i],
                    );
                }
                let first_camera = point % (cams - window + 1);
                for slot in 0..window {
                    let camera = first_camera + slot;
                    for i in 0..6 {
                        let index = camera * 6 + i;
                        g.set_at(
                            index,
                            g.at(index)
                                + contribution.camera_gradient[(point * window + slot) * 6 + i],
                        );
                    }
                }
            }
        }
        if let Some(h) = h {
            for point in 0..points {
                let mut point_block = h.block_mut::<3, 3>(point, point);
                for i in 0..3 {
                    for j in 0..3 {
                        point_block.set_at(
                            i,
                            j,
                            point_block.at(i, j)
                                + contribution.point_hessian[point * 9 + i * 3 + j],
                        );
                    }
                }
                let first_camera = point % (cams - window + 1);
                for slot in 0..window {
                    let camera = first_camera + slot;
                    let mut camera_block = h.block_mut::<6, 6>(camera, camera);
                    for i in 0..6 {
                        for j in 0..6 {
                            camera_block.set_at(
                                i,
                                j,
                                camera_block.at(i, j)
                                    + contribution.camera_hessian
                                        [(point * window + slot) * 36 + i * 6 + j],
                            );
                        }
                    }
                    let mut cross_block = h.block_mut::<6, 3>(camera, point);
                    for i in 0..6 {
                        for j in 0..3 {
                            cross_block.set_at(
                                i,
                                j,
                                contribution.cross[(point * window + slot) * 18 + i * 3 + j],
                            );
                        }
                    }
                }
            }
        }
        Cost::with_num_residuals(0.5 * cost, (2 * problem.observations.len()) as i32)
    };
    let summary = opt.optimize(&mut x, acc);
    outcome(&summary)
}
