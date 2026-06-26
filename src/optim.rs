use crate::tensor::{Node, TensorData};
use crate::device::EngineDevice;
use rayon::prelude::*;

// ========================================================================
// 1. LEARNING RATE SCHEDULERS
// ========================================================================
pub trait LRScheduler {
    fn get_lr(&self, step: u32) -> f32;
    fn step(&mut self, optimizer: &mut AdamW, step: u32) {
        let new_lr = self.get_lr(step);
        optimizer.set_learning_rate(new_lr);
    }
}

pub struct CosineAnnealingLR {
    pub initial_lr: f32,
    pub t_max: u32,
    pub eta_min: f32,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, t_max: u32, eta_min: f32) -> Self {
        Self { initial_lr, t_max, eta_min }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: u32) -> f32 {
        let t = step % self.t_max;
        self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1.0 + (std::f32::consts::PI * (t as f32) / (self.t_max as f32)).cos())
    }
}

pub struct LinearWarmup {
    pub target_lr: f32,
    pub warmup_steps: u32,
}

impl LinearWarmup {
    pub fn new(target_lr: f32, warmup_steps: u32) -> Self {
        Self { target_lr, warmup_steps }
    }
}

impl LRScheduler for LinearWarmup {
    fn get_lr(&self, step: u32) -> f32 {
        if step >= self.warmup_steps {
            self.target_lr
        } else {
            self.target_lr * ((step as f32 + 1.0) / (self.warmup_steps as f32))
        }
    }
}

// ========================================================================
// 2. DYNAMIC GRADIENT SCALER (AMP)
// ========================================================================
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
            scale: 65536.0, // Initial FP16/BF16 friendly scale
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            successful_steps: 0,
        }
    }

    /// Scales the loss forward before `.backward()` is called
    pub fn scale(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// Checks for NaNs, steps the optimizer if safe, and dynamically adjusts the scale
    pub fn step(&mut self, optimizer: &mut AdamW, has_nans: bool) {
        if has_nans {
            println!("⚠️ GradScaler: NaN detected. Halving loss scale to {}", self.scale * self.backoff_factor);
            self.scale *= self.backoff_factor;
            self.successful_steps = 0;
            optimizer.zero_grad(); // Abort the step to save the model!
        } else {
            optimizer.loss_scale = self.scale; // Sync scale to the optimizer for unscaling
            optimizer.step();
            
            self.successful_steps += 1;
            if self.successful_steps >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.successful_steps = 0;
                println!("📈 GradScaler: Stability reached. Doubling loss scale to {}", self.scale);
            }
        }
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}

// ========================================================================
// 3. ADAM-W OPTIMIZER (WITH FUSED UN-SCALING SHADERS)
// ========================================================================

// The Uniform structure must be a multiple of 16 bytes for wgpu.
// With loss_scale included, we have exactly 8 elements (8 * 4 = 32 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamWArgs {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    loss_scale: f32, // Automatic Mixed Precision Scaling
    step: u32,
    size: u32,
}

pub struct AdamWState {
    pub m: TensorData, // First moment (Momentum)
    pub v: TensorData, // Second moment (Variance)
}

pub struct AdamW {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub loss_scale: f32, // AMP Scale Factor
    
    pub parameters: Vec<Node>, 
    pub states: Vec<AdamWState>, 
    pub step_count: u32,         
}

impl AdamW {
    pub fn new(learning_rate: f32, parameters: Vec<Node>) -> Self {
        AdamW { 
            learning_rate, 
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            loss_scale: 65536.0, // Default AMP loss scale
            parameters,
            states: Vec::new(),
            step_count: 0,
        }
    }

    pub fn set_learning_rate(&mut self, new_lr: f32) {
        self.learning_rate = new_lr;
    }

    pub fn zero_grad(&self) {
        for param in &self.parameters {
            let mut p = param.write().unwrap();
            p.grad = None;
        }
    }

    pub fn step(&mut self) {
        let clip_threshold = 1.0;
        self.step_count += 1;

        if self.states.is_empty() {
            for param in &self.parameters {
                let p = param.read().unwrap();
                let size = p.shape.iter().product::<usize>();
                
                let state = match &p.data {
                    TensorData::Cpu(_) => {
                        AdamWState {
                            m: TensorData::Cpu(vec![0.0; size]),
                            v: TensorData::Cpu(vec![0.0; size]),
                        }
                    },
                    TensorData::Gpu(_) => {
                        if let EngineDevice::Gpu { device, .. } = &p.device {
                            let buf_size = (size * 4) as wgpu::BufferAddress;
                            let m_buf = device.create_buffer(&wgpu::BufferDescriptor { 
                                label: Some("AdamW Momentum Buffer"), size: buf_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false 
                            });
                            let v_buf = device.create_buffer(&wgpu::BufferDescriptor { 
                                label: Some("AdamW Variance Buffer"), size: buf_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false 
                            });
                            AdamWState { m: TensorData::Gpu(m_buf), v: TensorData::Gpu(v_buf) }
                        } else { unreachable!() }
                    },
                    TensorData::Lazy(_) => panic!("Cannot initialize AdamW state for a Lazy node! Compile the graph first."),
                };
                self.states.push(state);
            }
        }

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

                w = w - args.lr * args.weight_decay * w;

                let new_m = args.beta1 * m[i] + (1.0 - args.beta1) * g;
                let new_v = args.beta2 * v[i] + (1.0 - args.beta2) * (g * g);
                m[i] = new_m;
                v[i] = new_v;

                let t = f32(args.step);
                let m_hat = new_m / (1.0 - pow(args.beta1, t));
                let v_hat = new_v / (1.0 - pow(args.beta2, t));

                weights[i] = w - args.lr * m_hat / (sqrt(v_hat) + args.eps);
            }
        ";

        for (idx, param) in self.parameters.iter().enumerate() {
            let mut p = param.write().unwrap();
            let p_ref = &mut *p; 
            let state = &mut self.states[idx]; 
            
            if let Some(grad_data) = &p_ref.grad {
                match (&mut p_ref.data, grad_data) {
                    (TensorData::Cpu(weights), TensorData::Cpu(grad)) => {
                        let m_data = if let TensorData::Cpu(m) = &mut state.m { m } else { unreachable!() };
                        let v_data = if let TensorData::Cpu(v) = &mut state.v { v } else { unreachable!() };
                        
                        let b1 = self.beta1; let b2 = self.beta2; let eps = self.eps;
                        let lr = self.learning_rate; let wd = self.weight_decay; let t = self.step_count as f32;
                        let scale = self.loss_scale;
                        
                        weights.par_iter_mut()
                            .zip(grad.par_iter())
                            .zip(m_data.par_iter_mut())
                            .zip(v_data.par_iter_mut())
                            .for_each(|(((w, g), m), v)| {
                                *w -= lr * wd * *w; 
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
                    (TensorData::Gpu(weight_buf), TensorData::Cpu(grad)) => {
                        let m_buf = if let TensorData::Gpu(b) = &state.m { b } else { unreachable!() };
                        let v_buf = if let TensorData::Gpu(b) = &state.v { b } else { unreachable!() };

                        if let EngineDevice::Gpu { device, queue } = &p_ref.device {
                            let size = grad.len() as u32;
                            let args = AdamWArgs { lr: self.learning_rate, beta1: self.beta1, beta2: self.beta2, eps: self.eps, weight_decay: self.weight_decay, loss_scale: self.loss_scale, step: self.step_count, size };
                            
                            let args_buf = device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("AdamW Uniform Args"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
                            });
                            queue.write_buffer(&args_buf, 0, bytemuck::bytes_of(&args));

                            let grad_size = (grad.len() * 4) as wgpu::BufferAddress;
                            let grad_buf = device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("AdamW Temp Gradient"), size: grad_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
                            });
                            queue.write_buffer(&grad_buf, 0, bytemuck::cast_slice(grad));

                            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                                label: Some("AdamW Optimizer Shader"),
                                source: wgpu::ShaderSource::Wgsl(adamw_shader_source.into()),
                            });

                            let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                label: None,
                                entries: &[
                                    wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                                    wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                                    wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                                    wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                                    wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                                ],
                            });

                            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bind_group_layout)], immediate_size: 0 });
                            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("AdamW Pipeline"), layout: Some(&pipeline_layout), module: &shader, entry_point: Some("main"), cache: None, compilation_options: Default::default() });

                            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: None, layout: &bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry { binding: 0, resource: args_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 1, resource: weight_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 2, resource: grad_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 3, resource: m_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 4, resource: v_buf.as_entire_binding() },
                                ],
                            });

                            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                            {
                                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                                cpass.set_pipeline(&pipeline);
                                cpass.set_bind_group(0, &bind_group, &[]);
                                cpass.dispatch_workgroups(size.div_ceil(256), 1, 1);
                            }
                            queue.submit(Some(encoder.finish()));
                        } else { unreachable!() }
                    }
                    (TensorData::Gpu(weight_buf), TensorData::Gpu(grad_buf)) => {
                        let m_buf = if let TensorData::Gpu(b) = &state.m { b } else { unreachable!() };
                        let v_buf = if let TensorData::Gpu(b) = &state.v { b } else { unreachable!() };

                        if let EngineDevice::Gpu { device, queue } = &p_ref.device {
                            let size = (weight_buf.size() / 4) as u32;
                            let args = AdamWArgs { lr: self.learning_rate, beta1: self.beta1, beta2: self.beta2, eps: self.eps, weight_decay: self.weight_decay, loss_scale: self.loss_scale, step: self.step_count, size };
                            
                            let args_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("AdamW Uniform Args"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                            queue.write_buffer(&args_buf, 0, bytemuck::bytes_of(&args));

                            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("AdamW Optimizer Shader"), source: wgpu::ShaderSource::Wgsl(adamw_shader_source.into()) });

                            let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                label: None,
                                entries: &[
                                    wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                                    wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                                    wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                                    wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                                    wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                                ],
                            });

                            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bind_group_layout)], immediate_size: 0 });
                            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("AdamW Pipeline"), layout: Some(&pipeline_layout), module: &shader, entry_point: Some("main"), cache: None, compilation_options: Default::default() });

                            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: None, layout: &bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry { binding: 0, resource: args_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 1, resource: weight_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 2, resource: grad_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 3, resource: m_buf.as_entire_binding() },
                                    wgpu::BindGroupEntry { binding: 4, resource: v_buf.as_entire_binding() },
                                ],
                            });

                            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                            {
                                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                                cpass.set_pipeline(&pipeline);
                                cpass.set_bind_group(0, &bind_group, &[]);
                                cpass.dispatch_workgroups(size.div_ceil(256), 1, 1);
                            }
                            queue.submit(Some(encoder.finish()));
                        } else { unreachable!() }
                    }
                    (TensorData::Lazy(_), _) => panic!("Cannot step AdamW on Lazy node!"),
                    (_, TensorData::Lazy(_)) => panic!("Cannot step AdamW with Lazy gradient!"),
                    _ => panic!("Hardware deployment conflict"),
                }
            }
        }
    }
}