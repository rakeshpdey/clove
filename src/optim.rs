/*
 * src/attention.rs
 * Hardware considerations & Optimizer.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Rakesh Pradip Dey
 *
 * Licensed under the MIT License <LICENSE-MIT or http://opensource.org/licenses/MIT>.
 *
 * Note: Portions of this software are adapted from existing open-source frameworks.
 * This file may not be copied, modified, or distributed except according to the terms
 * of the MIT license.
 */

//! Optimizer and Learning Rate Scheduler module for Clove.
//!
//! This module implements essential training utilities, including common
//! learning rate scheduling strategies and a robust Automatic Mixed
//! Precision (AMP) Gradient Scaler.
//!
//! The core component is the `AdamW` optimizer, which utilizes a dual-backend
//! architecture to support both parallel CPU execution (via Rayon) and
//! high-throughput GPU compute shaders (via `wgpu`).

use crate::device::EngineDevice;
use crate::tensor::{Node, TensorData};
use rayon::prelude::*;

// LEARNING RATE SCHEDULERS

/// A common trait for all learning rate schedulers.
/// Defines how the learning rate is calculated at a specific step and applied to the optimizer.
pub trait LRScheduler {
    /// Calculates the learning rate for the given training step.
    fn get_lr(&self, step: u32) -> f32;

    /// Updates the optimizer's learning rate in place.
    fn step(&mut self, optimizer: &mut AdamW, step: u32) {
        let new_lr = self.get_lr(step);
        optimizer.set_learning_rate(new_lr);
    }
}

/// Cosine Annealing Learning Rate Scheduler.
/// Decays the learning rate using a cosine curve, smoothly dropping from `initial_lr`
/// to `eta_min` over `t_max` steps, and then repeating.    
pub struct CosineAnnealingLR {
    pub initial_lr: f32,
    pub t_max: u32,
    pub eta_min: f32,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, t_max: u32, eta_min: f32) -> Self {
        Self {
            initial_lr,
            t_max,
            eta_min,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: u32) -> f32 {
        // Calculate where we are in the current cycle
        let t = step % self.t_max;

        // Standard Cosine Annealing formula:
        // eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * t / t_max))
        self.eta_min
            + 0.5
                * (self.initial_lr - self.eta_min)
                * (1.0 + (std::f32::consts::PI * (t as f32) / (self.t_max as f32)).cos())
    }
}

/// Linear Warmup Scheduler.
/// Gradually increases the learning rate from 0 to `target_lr` linearly over a set
/// number of `warmup_steps`. Useful for preventing massive gradient spikes early in training.
pub struct LinearWarmup {
    pub target_lr: f32,
    pub warmup_steps: u32,
}

impl LinearWarmup {
    pub fn new(target_lr: f32, warmup_steps: u32) -> Self {
        Self {
            target_lr,
            warmup_steps,
        }
    }
}

impl LRScheduler for LinearWarmup {
    fn get_lr(&self, step: u32) -> f32 {
        if step >= self.warmup_steps {
            // Warmup is over, hold at target_lr
            self.target_lr
        } else {
            // Linearly interpolate towards target_lr
            self.target_lr * ((step as f32 + 1.0) / (self.warmup_steps as f32))
        }
    }
}

// DYNAMIC GRADIENT SCALER (AMP)

/// Handles Automatic Mixed Precision (AMP) gradient scaling.
/// Multiplies the loss by a large scale factor before backward pass to prevent
/// FP16/BF16 underflow. Dynamically adjusts the scale if NaNs are detected.
pub struct GradScaler {
    pub scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: u32,
    pub successful_steps: u32,
}

impl GradScaler {
    pub fn new() -> Self {
        Self {
            // Default initial scale (2^16), friendly for FP16/BF16
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            // Number of stable steps before increasing the scale
            growth_interval: 2000,
            successful_steps: 0,
        }
    }

    /// Scales the loss forward before `.backward()` is called
    /// This ensures gradients don't vanish into zeroes during backprop.
    pub fn scale(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// Checks for NaNs, steps the optimizer if safe, and dynamically adjusts the scale
    /// and drops the scale. Otherwise, it executes the step and tracks stability.
    pub fn step(&mut self, optimizer: &mut AdamW, has_nans: bool) {
        if has_nans {
            // Gradients exploded/NaN'd out. Penalize the scale factor.
            println!(
                "GradScaler: NaN detected. Halving loss scale to {}",
                self.scale * self.backoff_factor
            );
            self.scale *= self.backoff_factor;
            self.successful_steps = 0;

            // Note: Aborting the optimizer step saves the weights from becoming NaN
            optimizer.zero_grad();
        } else {
            // Safe to proceed. Feed the scale to the optimizer so it can unscale internally.
            optimizer.loss_scale = self.scale;
            optimizer.step();

            self.successful_steps += 1;
            // If we've survived `growth_interval` steps without NaNs, cautiously increase the scale.
            if self.successful_steps >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.successful_steps = 0;
                println!(
                    "GradScaler: Stability reached. Doubling loss scale to {}",
                    self.scale
                );
            }
        }
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}

// 3. ADAM-W OPTIMIZER (WITH FUSED UN-SCALING SHADERS)

/// Uniform buffer arguments sent to the GPU.
/// NOTE: The Uniform structure must be a multiple of 16 bytes for `wgpu` alignment rules.
/// Currently, we have 8 `f32` (or `u32`) elements: 8 * 4 = 32 bytes, which satisfies this perfectly.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamWArgs {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    loss_scale: f32,
    step: u32,
    size: u32,
}

/// Holds the momentum, variance, and hardware cache state for a single parameter tensor.
pub struct AdamWState {
    pub m: TensorData, // First moment (Momentum)
    pub v: TensorData, // Second moment (Variance)

    // Hardware Caches to prevent memory thrashing
    // Storing these prevents wgpu from thrashing memory by recreating buffers and bind groups every step.
    pub args_buf: Option<std::sync::Arc<wgpu::Buffer>>,
    pub bind_group: Option<std::sync::Arc<wgpu::BindGroup>>,
}

/// The AdamW Optimizer.
/// Includes support for CPU multithreading (rayon) and GPU compute shaders (wgpu).
pub struct AdamW {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub loss_scale: f32,

    pub parameters: Vec<Node>,
    pub states: Vec<AdamWState>,
    pub step_count: u32,

    // Hardware Caches to prevent shader re-compilation
    pub pipeline: Option<std::sync::Arc<wgpu::ComputePipeline>>,
    pub bind_group_layout: Option<std::sync::Arc<wgpu::BindGroupLayout>>,
}

impl AdamW {
    pub fn new(learning_rate: f32, parameters: Vec<Node>) -> Self {
        AdamW {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            loss_scale: 65536.0,
            parameters,
            states: Vec::new(),
            step_count: 0,
            pipeline: None,
            bind_group_layout: None,
        }
    }

    pub fn set_learning_rate(&mut self, new_lr: f32) {
        self.learning_rate = new_lr;
    }

    /// Wipes the gradients for all registered parameters.
    pub fn zero_grad(&self) {
        for param in &self.parameters {
            let mut p = param.write().unwrap();
            p.grad = None;
        }
    }

    /// Executes one optimization step, updating all weights.
    pub fn step(&mut self) {
        let clip_threshold = 1.0;
        self.step_count += 1;

        // Initialization Block: Create M and V tensors on the first step
        if self.states.is_empty() {
            for param in &self.parameters {
                let p = param.read().unwrap();
                let size = p.shape.iter().product::<usize>();

                let state = match &p.data {
                    TensorData::Cpu(_) => AdamWState {
                        m: TensorData::Cpu(vec![0.0; size]),
                        v: TensorData::Cpu(vec![0.0; size]),
                        args_buf: None,
                        bind_group: None,
                    },
                    TensorData::Gpu(_) => {
                        if let EngineDevice::Gpu { device, .. } = &p.device {
                            let buf_size = (size * 4) as wgpu::BufferAddress;
                            // Pre-allocate GPU buffers for momentum and variance
                            let m_buf = device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("AdamW Momentum Buffer"),
                                size: buf_size,
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            let v_buf = device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("AdamW Variance Buffer"),
                                size: buf_size,
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            AdamWState {
                                m: TensorData::Gpu(m_buf),
                                v: TensorData::Gpu(v_buf),
                                args_buf: None,
                                bind_group: None,
                            }
                        } else {
                            unreachable!()
                        }
                    }
                    TensorData::Lazy(_) => panic!(
                        "Cannot initialize AdamW state for a Lazy node! Compile the graph first."
                    ),
                };
                self.states.push(state);
            }
        }

        // WGSL Compute Shader Definition
        // Note: We fuse unscaling, clipping, weight decay, and the Adam update into ONE pass.
        // This dramatically reduces memory bandwidth compared to doing them sequentially.
        let adamw_shader_source = "
            struct AdamWArgs {
                lr: f32,
                beta1: f32,
                beta2: f32,
                eps: f32,
                weight_decay: f32,
                loss_scale: f32,
                step: u32,
                size: u32,
            }
            @group(0) @binding(0) var<uniform> args: AdamWArgs;
            @group(0) @binding(1) var<storage, read_write> weights: array<f32>;
            @group(0) @binding(2) var<storage, read> grads: array<f32>;
            @group(0) @binding(3) var<storage, read_write> m: array<f32>;
            @group(0) @binding(4) var<storage, read_write> v: array<f32>;

            @compute @workgroup_size(256, 1, 1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let i = id.x;
                if (i >= args.size) { return; }

                // AMP Unscaling & Clipping!
                let raw_g = grads[i] / args.loss_scale;
                let g = clamp(raw_g, -1.0, 1.0);
                var w = weights[i];

                // Weight Decay (Decoupled from gradients as per AdamW spec)
                w = w - args.lr * args.weight_decay * w;

                // Update biased first & second moments
                let new_m = args.beta1 * m[i] + (1.0 - args.beta1) * g;
                let new_v = args.beta2 * v[i] + (1.0 - args.beta2) * (g * g);
                m[i] = new_m;
                v[i] = new_v;

                // Bias correction
                let t = f32(args.step);
                let m_hat = new_m / (1.0 - pow(args.beta1, t));
                let v_hat = new_v / (1.0 - pow(args.beta2, t));

                // Final weight update
                weights[i] = w - args.lr * m_hat / (sqrt(v_hat) + args.eps);
            }
        ";

        // Execution Block
        for (idx, param) in self.parameters.iter().enumerate() {
            let mut p = param.write().unwrap();
            let p_ref = &mut *p;
            let state = &mut self.states[idx];

            if let Some(grad_data) = &p_ref.grad {
                match (&mut p_ref.data, grad_data) {
                    // Pure CPU Execution (Parallelized via Rayon)
                    (TensorData::Cpu(weights), TensorData::Cpu(grad)) => {
                        let m_data = if let TensorData::Cpu(m) = &mut state.m {
                            m
                        } else {
                            unreachable!()
                        };
                        let v_data = if let TensorData::Cpu(v) = &mut state.v {
                            v
                        } else {
                            unreachable!()
                        };

                        let b1 = self.beta1;
                        let b2 = self.beta2;
                        let eps = self.eps;
                        let lr = self.learning_rate;
                        let wd = self.weight_decay;
                        let t = self.step_count as f32;
                        let scale = self.loss_scale;

                        // par_iter_mut() heavily utilizes multi-core processors
                        weights
                            .par_iter_mut()
                            .zip(grad.par_iter())
                            .zip(m_data.par_iter_mut())
                            .zip(v_data.par_iter_mut())
                            .for_each(|(((w, g), m), v)| {
                                *w -= lr * wd * *w; // AdamW Weight Decay

                                // AMP UNSCALING ON CPU!
                                let g_unscaled = g / scale;
                                let clipped_g = g_unscaled.clamp(-clip_threshold, clip_threshold);

                                *m = b1 * *m + (1.0 - b1) * clipped_g;
                                *v = b2 * *v + (1.0 - b2) * clipped_g * clipped_g;

                                let m_hat = *m / (1.0 - b1.powf(t));
                                let v_hat = *v / (1.0 - b2.powf(t));

                                *w -= lr * m_hat / (v_hat.sqrt() + eps);
                            });
                    }

                    // CPU Gradients to GPU Weights
                    // (Sub-optimal, incurs host-to-device copy penalty)
                    (TensorData::Gpu(weight_buf), TensorData::Cpu(grad)) => {
                        let m_buf = if let TensorData::Gpu(b) = &state.m {
                            b
                        } else {
                            unreachable!()
                        };
                        let v_buf = if let TensorData::Gpu(b) = &state.v {
                            b
                        } else {
                            unreachable!()
                        };

                        if let EngineDevice::Gpu { device, queue } = &p_ref.device {
                            let size = grad.len() as u32;
                            let args = AdamWArgs {
                                lr: self.learning_rate,
                                beta1: self.beta1,
                                beta2: self.beta2,
                                eps: self.eps,
                                weight_decay: self.weight_decay,
                                loss_scale: self.loss_scale,
                                step: self.step_count,
                                size,
                            };

                            // Compile Pipeline Once
                            if self.pipeline.is_none() {
                                let shader =
                                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                                        label: Some("AdamW Optimizer Shader"),
                                        source: wgpu::ShaderSource::Wgsl(
                                            adamw_shader_source.into(),
                                        ),
                                    });
                                let bind_group_layout = device.create_bind_group_layout(
                                    &wgpu::BindGroupLayoutDescriptor {
                                        label: None,
                                        entries: &[
                                            wgpu::BindGroupLayoutEntry {
                                                binding: 0,
                                                visibility: wgpu::ShaderStages::COMPUTE,
                                                ty: wgpu::BindingType::Buffer {
                                                    ty: wgpu::BufferBindingType::Uniform,
                                                    has_dynamic_offset: false,
                                                    min_binding_size: None,
                                                },
                                                count: None,
                                            },
                                            wgpu::BindGroupLayoutEntry {
                                                binding: 1,
                                                visibility: wgpu::ShaderStages::COMPUTE,
                                                ty: wgpu::BindingType::Buffer {
                                                    ty: wgpu::BufferBindingType::Storage {
                                                        read_only: false,
                                                    },
                                                    has_dynamic_offset: false,
                                                    min_binding_size: None,
                                                },
                                                count: None,
                                            },
                                            wgpu::BindGroupLayoutEntry {
                                                binding: 2,
                                                visibility: wgpu::ShaderStages::COMPUTE,
                                                ty: wgpu::BindingType::Buffer {
                                                    ty: wgpu::BufferBindingType::Storage {
                                                        read_only: true,
                                                    },
                                                    has_dynamic_offset: false,
                                                    min_binding_size: None,
                                                },
                                                count: None,
                                            },
                                            wgpu::BindGroupLayoutEntry {
                                                binding: 3,
                                                visibility: wgpu::ShaderStages::COMPUTE,
                                                ty: wgpu::BindingType::Buffer {
                                                    ty: wgpu::BufferBindingType::Storage {
                                                        read_only: false,
                                                    },
                                                    has_dynamic_offset: false,
                                                    min_binding_size: None,
                                                },
                                                count: None,
                                            },
                                            wgpu::BindGroupLayoutEntry {
                                                binding: 4,
                                                visibility: wgpu::ShaderStages::COMPUTE,
                                                ty: wgpu::BindingType::Buffer {
                                                    ty: wgpu::BufferBindingType::Storage {
                                                        read_only: false,
                                                    },
                                                    has_dynamic_offset: false,
                                                    min_binding_size: None,
                                                },
                                                count: None,
                                            },
                                        ],
                                    },
                                );
                                let pipeline_layout = device.create_pipeline_layout(
                                    &wgpu::PipelineLayoutDescriptor {
                                        label: None,
                                        bind_group_layouts: &[Some(&bind_group_layout)],
                                        immediate_size: 0,
                                    },
                                );
                                let pipeline = device.create_compute_pipeline(
                                    &wgpu::ComputePipelineDescriptor {
                                        label: Some("AdamW Pipeline"),
                                        layout: Some(&pipeline_layout),
                                        module: &shader,
                                        entry_point: Some("main"),
                                        cache: None,
                                        compilation_options: Default::default(),
                                    },
                                );
                                self.bind_group_layout =
                                    Some(std::sync::Arc::new(bind_group_layout));
                                self.pipeline = Some(std::sync::Arc::new(pipeline));
                            }

                            // Initialize the args uniform buffer per-parameter if not present
                            if state.args_buf.is_none() {
                                state.args_buf = Some(std::sync::Arc::new(device.create_buffer(
                                    &wgpu::BufferDescriptor {
                                        label: Some("AdamW Uniform Args"),
                                        size: 32,
                                        usage: wgpu::BufferUsages::UNIFORM
                                            | wgpu::BufferUsages::COPY_DST,
                                        mapped_at_creation: false,
                                    },
                                )));
                            }
                            let args_buf = state.args_buf.as_ref().unwrap();

                            // Update uniform arguments (LR, step count, etc. change every tick)
                            queue.write_buffer(args_buf, 0, bytemuck::bytes_of(&args));

                            let grad_size = (grad.len() * 4) as wgpu::BufferAddress;
                            let grad_buf = device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("AdamW Temp Gradient"),
                                size: grad_size,
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            queue.write_buffer(&grad_buf, 0, bytemuck::cast_slice(grad));

                            // Note: Because grad_buf changes every step in this branch, we must recreate the bind group
                            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: None,
                                layout: self.bind_group_layout.as_ref().unwrap(),
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::Buffer(
                                            wgpu::BufferBinding {
                                                buffer: unsafe {
                                                    &*std::sync::Arc::as_ptr(args_buf)
                                                },
                                                offset: 0,
                                                size: None,
                                            },
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: weight_buf.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: grad_buf.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: m_buf.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 4,
                                        resource: v_buf.as_entire_binding(),
                                    },
                                ],
                            });

                            // Dispatch shader execution
                            let mut encoder =
                                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: None,
                                });
                            {
                                let mut cpass =
                                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                        label: None,
                                        timestamp_writes: None,
                                    });
                                cpass.set_pipeline(self.pipeline.as_ref().unwrap());
                                cpass.set_bind_group(0, &bind_group, &[]);
                                cpass.dispatch_workgroups(size.div_ceil(256), 1, 1);
                            }
                            queue.submit(Some(encoder.finish()));
                        } else {
                            unreachable!()
                        }
                    }

                    // Pure GPU Execution
                    (TensorData::Gpu(weight_buf), TensorData::Gpu(grad_buf)) => {
                        // FASTEST EXECUTION PATH: 100% Cached
                        let m_buf = if let TensorData::Gpu(b) = &state.m {
                            b
                        } else {
                            unreachable!()
                        };
                        let v_buf = if let TensorData::Gpu(b) = &state.v {
                            b
                        } else {
                            unreachable!()
                        };

                        if let EngineDevice::Gpu { device, queue } = &p_ref.device {
                            let size = (weight_buf.size() / 4) as u32;
                            let args = AdamWArgs {
                                lr: self.learning_rate,
                                beta1: self.beta1,
                                beta2: self.beta2,
                                eps: self.eps,
                                weight_decay: self.weight_decay,
                                loss_scale: self.loss_scale,
                                step: self.step_count,
                                size,
                            };

                            // Compile Pipeline Once
                            if self.pipeline.is_none() {
                                let shader =
                                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                                        label: Some("AdamW Optimizer Shader"),
                                        source: wgpu::ShaderSource::Wgsl(
                                            adamw_shader_source.into(),
                                        ),
                                    });
                                let bind_group_layout = device.create_bind_group_layout(
                                    &wgpu::BindGroupLayoutDescriptor {
                                        label: None,
                                        entries: &[
                                            wgpu::BindGroupLayoutEntry {
                                                binding: 0,
                                                visibility: wgpu::ShaderStages::COMPUTE,
                                                ty: wgpu::BindingType::Buffer {
                                                    ty: wgpu::BufferBindingType::Uniform,
                                                    has_dynamic_offset: false,
                                                    min_binding_size: None,
                                                },
                                                count: None,
                                            },
                                            wgpu::BindGroupLayoutEntry {
                                                binding: 1,
                                                visibility: wgpu::ShaderStages::COMPUTE,
                                                ty: wgpu::BindingType::Buffer {
                                                    ty: wgpu::BufferBindingType::Storage {
                                                        read_only: false,
                                                    },
                                                    has_dynamic_offset: false,
                                                    min_binding_size: None,
                                                },
                                                count: None,
                                            },
                                            wgpu::BindGroupLayoutEntry {
                                                binding: 2,
                                                visibility: wgpu::ShaderStages::COMPUTE,
                                                ty: wgpu::BindingType::Buffer {
                                                    ty: wgpu::BufferBindingType::Storage {
                                                        read_only: true,
                                                    },
                                                    has_dynamic_offset: false,
                                                    min_binding_size: None,
                                                },
                                                count: None,
                                            },
                                            wgpu::BindGroupLayoutEntry {
                                                binding: 3,
                                                visibility: wgpu::ShaderStages::COMPUTE,
                                                ty: wgpu::BindingType::Buffer {
                                                    ty: wgpu::BufferBindingType::Storage {
                                                        read_only: false,
                                                    },
                                                    has_dynamic_offset: false,
                                                    min_binding_size: None,
                                                },
                                                count: None,
                                            },
                                            wgpu::BindGroupLayoutEntry {
                                                binding: 4,
                                                visibility: wgpu::ShaderStages::COMPUTE,
                                                ty: wgpu::BindingType::Buffer {
                                                    ty: wgpu::BufferBindingType::Storage {
                                                        read_only: false,
                                                    },
                                                    has_dynamic_offset: false,
                                                    min_binding_size: None,
                                                },
                                                count: None,
                                            },
                                        ],
                                    },
                                );
                                let pipeline_layout = device.create_pipeline_layout(
                                    &wgpu::PipelineLayoutDescriptor {
                                        label: None,
                                        bind_group_layouts: &[Some(&bind_group_layout)],
                                        immediate_size: 0,
                                    },
                                );
                                let pipeline = device.create_compute_pipeline(
                                    &wgpu::ComputePipelineDescriptor {
                                        label: Some("AdamW Pipeline"),
                                        layout: Some(&pipeline_layout),
                                        module: &shader,
                                        entry_point: Some("main"),
                                        cache: None,
                                        compilation_options: Default::default(),
                                    },
                                );
                                self.bind_group_layout =
                                    Some(std::sync::Arc::new(bind_group_layout));
                                self.pipeline = Some(std::sync::Arc::new(pipeline));
                            }

                            // Cache Uniform Buffer (Write only to update variables)
                            if state.args_buf.is_none() {
                                state.args_buf = Some(std::sync::Arc::new(device.create_buffer(
                                    &wgpu::BufferDescriptor {
                                        label: Some("AdamW Uniform Args"),
                                        size: 32,
                                        usage: wgpu::BufferUsages::UNIFORM
                                            | wgpu::BufferUsages::COPY_DST,
                                        mapped_at_creation: false,
                                    },
                                )));
                            }
                            let args_buf = state.args_buf.as_ref().unwrap();
                            // Overwrite buffer data extremely quickly (Step, LR, Loss scale)
                            queue.write_buffer(args_buf, 0, bytemuck::bytes_of(&args));

                            // Cache Bind Group (Reused completely every step!)
                            // Reused completely every step! Because buffers are persistent in this path
                            // we avoid constructing bindings over and over.
                            if state.bind_group.is_none() {
                                state.bind_group = Some(std::sync::Arc::new(
                                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: None,
                                        layout: self.bind_group_layout.as_ref().unwrap(),
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: wgpu::BindingResource::Buffer(
                                                    wgpu::BufferBinding {
                                                        buffer: unsafe {
                                                            &*std::sync::Arc::as_ptr(args_buf)
                                                        },
                                                        offset: 0,
                                                        size: None,
                                                    },
                                                ),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 1,
                                                resource: weight_buf.as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 2,
                                                resource: grad_buf.as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 3,
                                                resource: m_buf.as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 4,
                                                resource: v_buf.as_entire_binding(),
                                            },
                                        ],
                                    }),
                                ));
                            }

                            // Dispatch shader execution
                            let mut encoder =
                                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: None,
                                });
                            {
                                let mut cpass =
                                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                        label: None,
                                        timestamp_writes: None,
                                    });
                                cpass.set_pipeline(self.pipeline.as_ref().unwrap());

                                // Use .as_deref() so Option<Arc<BindGroup>> resolves precisely to &BindGroup
                                cpass.set_bind_group(0, state.bind_group.as_deref().unwrap(), &[]);

                                cpass.dispatch_workgroups(size.div_ceil(256), 1, 1);
                            }
                            queue.submit(Some(encoder.finish()));
                        } else {
                            unreachable!()
                        }
                    }

                    // Invalid States (Lazy evaluation)
                    (TensorData::Lazy(_), _) => panic!("Cannot step AdamW on Lazy node!"),
                    (_, TensorData::Lazy(_)) => panic!("Cannot step AdamW with Lazy gradient!"),
                    _ => panic!("Hardware deployment conflict"),
                }
            }
        }
    }
}
