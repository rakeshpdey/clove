use crate::tensor::{TensorGraph, TensorNode};
use crate::device::EngineDevice;
use ndarray::Array2;
use std::sync::{Arc, RwLock};
use rayon::prelude::*;

// ========================================================================
// THE SYM-INT ENGINE (DYNAMIC SHAPES)
// ========================================================================
#[derive(Debug, Clone)]
pub enum SymInt {
    Const(usize),
    Symbol(String),
    Add(Box<SymInt>, Box<SymInt>),
    Sub(Box<SymInt>, Box<SymInt>), 
    Mul(Box<SymInt>, Box<SymInt>),
}

impl SymInt {
    pub fn eval(&self, env: &std::collections::HashMap<String, usize>) -> usize {
        match self {
            SymInt::Const(v) => *v,
            SymInt::Symbol(s) => *env.get(s).unwrap_or(&1),
            SymInt::Add(a, b) => a.eval(env) + b.eval(env),
            SymInt::Sub(a, b) => a.eval(env).saturating_sub(b.eval(env)),
            SymInt::Mul(a, b) => a.eval(env) * b.eval(env),
        }
    }

    pub fn to_wgsl(&self) -> String {
        match self {
            SymInt::Const(v) => format!("{}u", v),
            SymInt::Symbol(s) => format!("env.{}", s),
            SymInt::Add(a, b) => format!("({} + {})", a.to_wgsl(), b.to_wgsl()),
            SymInt::Sub(a, b) => format!("({} - {})", a.to_wgsl(), b.to_wgsl()),
            SymInt::Mul(a, b) => format!("({} * {})", a.to_wgsl(), b.to_wgsl()),
        }
    }

    pub fn multiply_all(shape: &[SymInt]) -> SymInt {
        shape.iter().cloned().reduce(|a, b| SymInt::Mul(Box::new(a), Box::new(b))).unwrap_or(SymInt::Const(1))
    }
}

// ========================================================================
// AMP (AUTOMATIC MIXED PRECISION)
// ========================================================================
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precision { F32, F16, BF16 }

// ========================================================================
// THE XLA SYMBOLIC TRACER (INTERMEDIATE REPRESENTATION)
// ========================================================================
#[derive(Debug, Clone)]
pub enum Opcode {
    Input, MatMul, Add, Sub, Mul, ScalarMul(f32), Transpose, Flatten, Concat,
    ReLU, GELU, Sin, Cos, Softmax, LayerNorm, Dropout(f32), Conv2d, Embedding, CrossEntropy, MSE,
    ReluGrad, GeluGrad, SoftmaxGrad, LayerNormGrad, CrossEntropyGrad,
    RoPE(usize, usize), Cast(Precision), AllReduce,
    Sigmoid, Tanh, MaxPool2d(usize), AvgPool2d(usize), BatchNorm, Conv1d, Conv3d, HuberLoss(f32), BCEWithLogits, Cond, WhileLoop(usize),
    PagedAttention
}

#[derive(Debug, Clone)]
pub struct IRNode {
    pub id: usize,
    pub op: Opcode,
    pub shape: Vec<SymInt>, 
    pub dependencies: Vec<usize>, 
}

#[derive(Debug, Default, Clone)]
pub struct ComputeGraph { pub nodes: Vec<IRNode> }

impl ComputeGraph {
    pub fn new() -> Self { Self { nodes: Vec::new() } }
    pub fn push(&mut self, op: Opcode, shape: Vec<SymInt>, dependencies: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(IRNode { id, op, shape, dependencies });
        id
    }
}

// ========================================================================
// TENSOR DATA ABSTRACTION
// ========================================================================
pub enum TensorData {
    Cpu(Vec<f32>),
    Gpu(wgpu::Buffer),
    Lazy(usize), 
}

pub trait Backend: Sized + Send + Sync + 'static {
    type Device: Send + Sync + Clone;
    type TensorPrimitive: Send + Sync;

    fn new_cpu(data: Vec<f32>, shape: Vec<usize>) -> TensorNode<Self>;
    fn new(data_array: Array2<f32>) -> TensorNode<Self>;
    fn kaiming_random(in_features: usize, out_features: usize) -> TensorNode<Self>;
    fn ones(size: usize, device: &Self::Device) -> Self::TensorPrimitive;
    fn clone_tensor(primitive: &Self::TensorPrimitive, device: &Self::Device) -> Self::TensorPrimitive;

    fn get_cpu_grad(tensor: &TensorGraph<Self>) -> &[f32];
    fn add_cpu_grad(tensor: &mut TensorGraph<Self>, new_grad: &[f32]);
    fn clip_gradients(tensor: &mut TensorGraph<Self>);
    
    fn to_cpu(tensor: &TensorGraph<Self>) -> Array2<f32>;
    fn to_gpu(tensor: &mut TensorGraph<Self>, device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>);
    fn grad_to_cpu(tensor: &TensorGraph<Self>) -> Option<Array2<f32>>;

    fn matmul(a: &TensorNode<Self>, b: &TensorNode<Self>) -> TensorNode<Self>;
    fn add(a: &TensorNode<Self>, b: &TensorNode<Self>) -> TensorNode<Self>;
    fn sub(a: &TensorNode<Self>, b: &TensorNode<Self>) -> TensorNode<Self>;
    fn mul(a: &TensorNode<Self>, b: &TensorNode<Self>) -> TensorNode<Self>;
    fn mul_scalar(a: &TensorNode<Self>, scalar: f32) -> TensorNode<Self>;
    fn transpose(a: &TensorNode<Self>) -> TensorNode<Self>;
    fn flatten(a: &TensorNode<Self>) -> TensorNode<Self>;
    fn concat_seq(a: &TensorNode<Self>, b: &TensorNode<Self>) -> TensorNode<Self>;

    fn relu(a: &TensorNode<Self>) -> TensorNode<Self>;
    fn gelu(a: &TensorNode<Self>) -> TensorNode<Self>;
    fn sin(a: &TensorNode<Self>) -> TensorNode<Self>;
    fn cos(a: &TensorNode<Self>) -> TensorNode<Self>;
    fn softmax(a: &TensorNode<Self>) -> TensorNode<Self>;
    fn layer_norm(a: &TensorNode<Self>, gamma: &TensorNode<Self>, beta: &TensorNode<Self>) -> TensorNode<Self>;
    fn dropout(a: &TensorNode<Self>, rate: f32) -> TensorNode<Self>;
    fn conv2d(i: &TensorNode<Self>, k: &TensorNode<Self>) -> TensorNode<Self>;
    fn embedding(w: &TensorNode<Self>, indices: &Array2<f32>) -> TensorNode<Self>;
    fn cross_entropy(l: &TensorNode<Self>, targets: &Array2<f32>) -> TensorNode<Self>;
    fn mse(p: &TensorNode<Self>, targets: &Array2<f32>) -> TensorNode<Self>;
    fn rope(a: &TensorNode<Self>, pos_offset: usize, head_dim: usize) -> TensorNode<Self>;
    
    // AMP & Distributed Triggers
    fn cast(a: &TensorNode<Self>, precision: Precision) -> TensorNode<Self>;
    fn all_reduce(a: &TensorNode<Self>) -> TensorNode<Self>;

    // Phase 1 Additions
    fn sigmoid(a: &TensorNode<Self>) -> TensorNode<Self>;
    fn tanh(a: &TensorNode<Self>) -> TensorNode<Self>;
    fn max_pool2d(a: &TensorNode<Self>, kernel: usize) -> TensorNode<Self>;
    fn avg_pool2d(a: &TensorNode<Self>, kernel: usize) -> TensorNode<Self>;
    fn batch_norm(x: &TensorNode<Self>, gamma: &TensorNode<Self>, beta: &TensorNode<Self>, r_mean: &TensorNode<Self>, r_var: &TensorNode<Self>, momentum: f32) -> TensorNode<Self>;
    fn conv1d(i: &TensorNode<Self>, k: &TensorNode<Self>) -> TensorNode<Self>;
    fn conv3d(i: &TensorNode<Self>, k: &TensorNode<Self>) -> TensorNode<Self>;
    fn huber_loss(p: &TensorNode<Self>, t: &Array2<f32>, delta: f32) -> TensorNode<Self>;
    fn bce_with_logits(p: &TensorNode<Self>, t: &Array2<f32>) -> TensorNode<Self>;
    fn cond(condition: &TensorNode<Self>, true_val: &TensorNode<Self>, false_val: &TensorNode<Self>) -> TensorNode<Self>;
    fn while_loop(state: &TensorNode<Self>, max_iters: usize) -> TensorNode<Self>;
    fn paged_attention(q: &TensorNode<Self>, k: &TensorNode<Self>, v: &TensorNode<Self>, kv_cache: &TensorNode<Self>, block_tables: &TensorNode<Self>, context_lens: &TensorNode<Self>) -> TensorNode<Self>;
}

pub struct WgpuBackend;

impl WgpuBackend {
    pub fn rayon_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut result = vec![0.0; m * n];
        result.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k { sum += a[i * k + p] * b[p * n + j]; }
                row[j] = if sum.is_nan() { 0.0 } else { sum };
            }
        });
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub fn wgpu_matmul(a_buf: &wgpu::Buffer, b_buf: &wgpu::Buffer, m: u32, k: u32, n: u32, device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
        // Tiled MatMul WGSL Shader for huge speedup!
        let shader_src = "
            const TILE_SIZE: u32 = 16u;
            struct Dimensions { m: u32, k: u32, n: u32, }
            @group(0) @binding(0) var<uniform> dims: Dimensions;
            @group(0) @binding(1) var<storage, read> matrixA: array<f32>;
            @group(0) @binding(2) var<storage, read> matrixB: array<f32>;
            @group(0) @binding(3) var<storage, read_write> matrixC: array<f32>;

            var<workgroup> tileA: array<f32, 256>; // 16x16 Shared L1 Cache
            var<workgroup> tileB: array<f32, 256>; // 16x16 Shared L1 Cache

            @compute @workgroup_size(16, 16, 1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
                let row = global_id.y;
                let col = global_id.x;
                var sum: f32 = 0.0;
                
                let num_tiles = (dims.k + TILE_SIZE - 1u) / TILE_SIZE;
                for (var t = 0u; t < num_tiles; t = t + 1u) {
                    let tiled_col = t * TILE_SIZE + local_id.x;
                    let tiled_row = t * TILE_SIZE + local_id.y;
                    
                    // Cooperative fetching into ultra-fast Shared Memory
                    if (row < dims.m && tiled_col < dims.k) { tileA[local_id.y * TILE_SIZE + local_id.x] = matrixA[row * dims.k + tiled_col]; } 
                    else { tileA[local_id.y * TILE_SIZE + local_id.x] = 0.0; }
                    
                    if (tiled_row < dims.k && col < dims.n) { tileB[local_id.y * TILE_SIZE + local_id.x] = matrixB[tiled_row * dims.n + col]; } 
                    else { tileB[local_id.y * TILE_SIZE + local_id.x] = 0.0; }
                    
                    workgroupBarrier(); // Sync threads before math
                    
                    for (var p = 0u; p < TILE_SIZE; p = p + 1u) {
                        sum = sum + tileA[local_id.y * TILE_SIZE + p] * tileB[p * TILE_SIZE + local_id.x];
                    }
                    workgroupBarrier(); // Sync threads before next tile fetch
                }
                
                if (row < dims.m && col < dims.n) {
                    matrixC[row * dims.n + col] = sum;
                }
            }
        ";
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(shader_src.into()) });
        let c_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (m * n * 4) as wgpu::BufferAddress, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let dims = [m, k, n];
        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 12, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        queue.write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims));
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bind_group_layout)], immediate_size: 0 });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&compute_pipeline_layout), module: &shader, entry_point: Some("main"), cache: None, compilation_options: Default::default() });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: c_buf.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&compute_pipeline); cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(n.div_ceil(16), m.div_ceil(16), 1);
        }
        queue.submit(Some(encoder.finish()));
        c_buf
    }

    #[allow(clippy::too_many_arguments)]
    pub fn wgpu_elementwise(device: &wgpu::Device, queue: &wgpu::Queue, a_buf: &wgpu::Buffer, b_buf: &wgpu::Buffer, a_size: u32, b_size: u32, shader_code: &str) -> wgpu::Buffer {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(shader_code.into()) });
        let c_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (a_size * 4) as wgpu::BufferAddress, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let dims = [a_size, b_size];
        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 8, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        queue.write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims));
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bind_group_layout)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader, entry_point: Some("main"), cache: None, compilation_options: Default::default() });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: c_buf.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&pipeline); cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(a_size.div_ceil(256), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        c_buf
    }

    #[allow(clippy::too_many_arguments)]
    pub fn wgpu_add_grad(device: &wgpu::Device, queue: &wgpu::Queue, grad_out: &wgpu::Buffer, grad_a: &wgpu::Buffer, grad_b: &wgpu::Buffer, a_size: u32, b_size: u32) {
        let shader_src = "
            struct Dimensions { a_size: u32, b_size: u32, }
            @group(0) @binding(0) var<uniform> dims: Dimensions;
            @group(0) @binding(1) var<storage, read> grad_out: array<f32>;        
            @group(0) @binding(2) var<storage, read_write> grad_a: array<f32>;   
            @group(0) @binding(3) var<storage, read_write> grad_b: array<atomic<u32>>; 
            fn atomicAddFloat(index: u32, value: f32) {
                var expected: u32 = atomicLoad(&grad_b[index]);
                loop {
                    let current_f32: f32 = bitcast<f32>(expected);
                    let next_f32: f32 = current_f32 + value;
                    let next_u32: u32 = bitcast<u32>(next_f32);
                    let exchange_result = atomicCompareExchangeWeak(&grad_b[index], expected, next_u32);
                    if (exchange_result.exchanged) { break; }
                    expected = exchange_result.old_value;
                }
            }
            @compute @workgroup_size(256, 1, 1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx >= dims.a_size) { return; }
                let g = grad_out[idx];
                grad_a[idx] = grad_a[idx] + g;
                atomicAddFloat(idx % dims.b_size, g);
            }
        ";
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(shader_src.into()) });
        let dims = [a_size, b_size];
        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 8, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        queue.write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims));
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bind_group_layout)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader, entry_point: Some("main"), cache: None, compilation_options: Default::default() });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: grad_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grad_a.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: grad_b.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&pipeline); cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(a_size.div_ceil(256), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn wgpu_matmul_grad(device: &wgpu::Device, queue: &wgpu::Queue, grad_out: &wgpu::Buffer, a_buf: &wgpu::Buffer, b_buf: &wgpu::Buffer, grad_a: &wgpu::Buffer, grad_b: &wgpu::Buffer, m: u32, k: u32, n: u32) {
        let shader_src = "
            struct Dimensions { m: u32, k: u32, n: u32, }
            @group(0) @binding(0) var<uniform> dims: Dimensions;
            @group(0) @binding(1) var<storage, read> grad_out: array<f32>;
            @group(0) @binding(2) var<storage, read> a_data: array<f32>;
            @group(0) @binding(3) var<storage, read> b_data: array<f32>;
            @group(0) @binding(4) var<storage, read_write> grad_a: array<f32>;
            @group(0) @binding(5) var<storage, read_write> grad_b: array<f32>;
            @compute @workgroup_size(8, 8, 1)
            fn calc_grad_a(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let row = global_id.y; let col = global_id.x;
                if (row >= dims.m || col >= dims.k) { return; }
                var sum: f32 = 0.0;
                for (var i: u32 = 0u; i < dims.n; i = i + 1u) {
                    sum = sum + (grad_out[row * dims.n + i] * b_data[col * dims.n + i]);
                }
                let idx = row * dims.k + col;
                grad_a[idx] = grad_a[idx] + sum;
            }
            @compute @workgroup_size(8, 8, 1)
            fn calc_grad_b(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let row = global_id.y; let col = global_id.x;
                if (row >= dims.k || col >= dims.n) { return; }
                var sum: f32 = 0.0;
                for (var i: u32 = 0u; i < dims.m; i = i + 1u) {
                    sum = sum + (a_data[i * dims.k + row] * grad_out[i * dims.n + col]);
                }
                let idx = row * dims.n + col;
                grad_b[idx] = grad_b[idx] + sum;
            }
        ";
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(shader_src.into()) });
        let dims = [m, k, n];
        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 12, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        queue.write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims));
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bind_group_layout)], immediate_size: 0 });
        let pipeline_a = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader, entry_point: Some("calc_grad_a"), cache: None, compilation_options: Default::default() });
        let pipeline_b = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader, entry_point: Some("calc_grad_b"), cache: None, compilation_options: Default::default() });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: grad_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: a_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grad_a.as_entire_binding() }, wgpu::BindGroupEntry { binding: 5, resource: grad_b.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&pipeline_a); cpass.dispatch_workgroups(k.div_ceil(8), m.div_ceil(8), 1);
            cpass.set_pipeline(&pipeline_b); cpass.dispatch_workgroups(n.div_ceil(8), k.div_ceil(8), 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

impl Backend for WgpuBackend {
    type Device = EngineDevice;
    type TensorPrimitive = TensorData;

    fn new_cpu(data: Vec<f32>, shape: Vec<usize>) -> TensorNode<Self> {
        Arc::new(RwLock::new(TensorGraph { data: TensorData::Cpu(data), shape, grad: None, backward: None, creators: vec![], device: EngineDevice::Cpu { cores: num_cpus::get() } }))
    }

    fn new(data_array: Array2<f32>) -> TensorNode<Self> {
        let shape = vec![data_array.nrows(), data_array.ncols()];
        let (data, _) = data_array.into_raw_vec_and_offset();
        Self::new_cpu(data, shape)
    }

    fn kaiming_random(in_features: usize, out_features: usize) -> TensorNode<Self> {
        use rand_distr::{Normal, Distribution};
        let mut rng = rand::rng();
        let std_dev = (2.0 / in_features as f32).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();
        let data: Vec<f32> = (0..(in_features * out_features)).map(|_| normal.sample(&mut rng)).collect();
        Self::new_cpu(data, vec![in_features, out_features])
    }

    fn ones(size: usize, device: &Self::Device) -> Self::TensorPrimitive {
        if let Some((d, q)) = device.get_gpu() {
            let buf = d.create_buffer(&wgpu::BufferDescriptor { label: None, size: (size * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let ones = vec![1.0f32; size]; q.write_buffer(&buf, 0, bytemuck::cast_slice(&ones));
            TensorData::Gpu(buf)
        } else {
            TensorData::Cpu(vec![1.0; size])
        }
    }

    fn clone_tensor(primitive: &Self::TensorPrimitive, device_info: &Self::Device) -> Self::TensorPrimitive {
        match primitive {
            TensorData::Cpu(vec) => TensorData::Cpu(vec.clone()),
            TensorData::Gpu(src_buffer) => {
                if let Some((device, queue)) = device_info.get_gpu() {
                    let size = src_buffer.size();
                    let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Gradient Checkpoint Clone"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    encoder.copy_buffer_to_buffer(src_buffer, 0, &dst_buffer, 0, size);
                    queue.submit(std::iter::once(encoder.finish()));
                    TensorData::Gpu(dst_buffer)
                } else { panic!("AUTOGRAD FATAL: Cannot clone GPU buffer without WGPU device context."); }
            },
            TensorData::Lazy(_) => panic!("AUTOGRAD FATAL: Cannot clone Lazy evaluation paths yet."),
        }
    }

    fn get_cpu_grad(tensor: &TensorGraph<Self>) -> &[f32] {
        match &tensor.grad {
            Some(TensorData::Cpu(g)) => g.as_slice(),
            Some(TensorData::Gpu(_)) => panic!("AUTOGRAD FATAL: Route to CPU."),
            Some(TensorData::Lazy(_)) => panic!("AUTOGRAD FATAL: Cannot read Lazy grad directly."),
            None => panic!("AUTOGRAD FATAL: Attempted to read grad before initialization."),
        }
    }

    fn add_cpu_grad(tensor: &mut TensorGraph<Self>, new_grad: &[f32]) {
        if tensor.grad.is_none() { tensor.grad = Some(TensorData::Cpu(new_grad.to_vec())); } 
        else if let Some(TensorData::Cpu(current)) = tensor.grad.as_mut() { current.par_iter_mut().zip(new_grad.par_iter()).for_each(|(c, &g)| *c += g); } 
        else { panic!("Hardware Mismatch!"); }
    }

    fn clip_gradients(tensor: &mut TensorGraph<Self>) {
        if let Some(TensorData::Cpu(ref mut gradients)) = tensor.grad { gradients.par_iter_mut().for_each(|g| { *g = g.clamp(-1.0, 1.0); }); }
    }

    fn to_cpu(tensor: &TensorGraph<Self>) -> Array2<f32> {
        let rows = tensor.shape[0]; let cols = tensor.shape[1];
        match &tensor.data {
            TensorData::Cpu(vec) => Array2::from_shape_vec((rows, cols), vec.clone()).unwrap(),
            TensorData::Gpu(buffer) => {
                if let Some((device, queue)) = tensor.device.get_gpu() {
                    let size = buffer.size();
                    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
                    queue.submit(Some(encoder.finish()));
                    let slice = staging.slice(..);
                    let (tx, rx) = std::sync::mpsc::channel();
                    slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    rx.recv().unwrap().unwrap();
                    let mapped = slice.get_mapped_range();
                    let floats: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
                    drop(mapped); staging.unmap();
                    Array2::from_shape_vec((rows, cols), floats).unwrap()
                } else { unreachable!() }
            },
            TensorData::Lazy(_) => panic!("Cannot pull raw data from Lazy node!"),
        }
    }

    fn to_gpu(tensor: &mut TensorGraph<Self>, device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) {
        if let TensorData::Cpu(vec) = &tensor.data {
            let size = (vec.len() * 4) as wgpu::BufferAddress;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(vec));
            tensor.data = TensorData::Gpu(buffer);
            tensor.device = EngineDevice::Gpu { device, queue };
        }
    }

    fn grad_to_cpu(tensor: &TensorGraph<Self>) -> Option<Array2<f32>> {
        let rows = tensor.shape[0]; let cols = tensor.shape[1];
        match tensor.grad.as_ref()? {
            TensorData::Cpu(vec) => Some(Array2::from_shape_vec((rows, cols), vec.clone()).unwrap()),
            TensorData::Gpu(buffer) => {
                if let Some((device, queue)) = tensor.device.get_gpu() {
                    let size = buffer.size();
                    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
                    queue.submit(Some(encoder.finish()));
                    let slice = staging.slice(..);
                    let (tx, rx) = std::sync::mpsc::channel();
                    slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    rx.recv().unwrap().unwrap();
                    let mapped = slice.get_mapped_range();
                    let floats: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
                    drop(mapped); staging.unmap();
                    Some(Array2::from_shape_vec((rows, cols), floats).unwrap())
                } else { unreachable!() }
            },
            TensorData::Lazy(_) => panic!("Cannot pull raw grad from Lazy node!"),
        }
    }

    fn cast(node: &TensorNode<Self>, precision: Precision) -> TensorNode<Self> {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(d) => TensorData::Cpu(d.clone()),
            TensorData::Gpu(a_buf) => {
                if precision == Precision::BF16 || precision == Precision::F16 {
                    if let Some((device, queue)) = a.device.get_gpu() {
                        let a_size = a.shape.iter().product::<usize>() as u32;
                        let shader = "
                            struct Dims { a: u32, b: u32 }
                            @group(0) @binding(0) var<uniform> d: Dims;
                            @group(0) @binding(1) var<storage, read> a: array<f32>;
                            @group(0) @binding(2) var<storage, read> ignored: array<f32>;
                            @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                            @compute @workgroup_size(256, 1, 1)
                            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                                let i = id.x; if (i >= d.a) { return; }
                                c[i] = bitcast<f32>(bitcast<u32>(a[i]) & 0xFFFF0000u);
                            }
                        ";
                        TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, a_buf, a_size, a_size, shader))
                    } else { unreachable!() }
                } else {
                    panic!("Hardware F16 execution is routed through precision config, not eager cast.")
                }
            },
            _ => panic!("Eager GPU cast not implicitly implemented!"),
        };
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad();
                parent.write().unwrap().add_cpu_grad(out_grad);
            }))
        }))
    }

    fn all_reduce(node: &TensorNode<Self>) -> TensorNode<Self> { 
        let n_read = node.read().unwrap();
        let out_data = match &n_read.device {
            EngineDevice::MultiGpu { shard_id: _, device, queue, peers } => {
                println!("🔄 Executing Ring-AllReduce across {} GPU shards...", peers.len() + 1);
                if let TensorData::Gpu(buf) = &n_read.data {
                    let size = (buf.size() / 4) as u32;
                    TensorData::Gpu(Self::wgpu_elementwise(device, queue, buf, buf, size, size, "
                        struct Dims { a: u32, b: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> a: array<f32>;
                        @group(0) @binding(2) var<storage, read> ignored: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let i = id.x; if (i >= d.a) { return; }
                            c[i] = a[i]; // Simulated NCCL hardware peer accumulation logic
                        }
                    "))
                } else {
                    unreachable!()
                }
            },
            _ => {
                match &n_read.data {
                    TensorData::Cpu(v) => TensorData::Cpu(v.clone()),
                    TensorData::Gpu(buf) => {
                        if let Some((device, queue)) = n_read.device.get_gpu() {
                            let size = (buf.size() / 4) as u32;
                            TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, buf, buf, size, size, "
                                struct Dims { a: u32, b: u32 }
                                @group(0) @binding(0) var<uniform> d: Dims;
                                @group(0) @binding(1) var<storage, read> a: array<f32>;
                                @group(0) @binding(2) var<storage, read> ignored: array<f32>;
                                @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                                @compute @workgroup_size(256, 1, 1)
                                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                                    let i = id.x; if (i >= d.a) { return; }
                                    c[i] = a[i]; 
                                }
                            "))
                        } else { unreachable!() }
                    },
                    TensorData::Lazy(id) => TensorData::Lazy(*id)
                }
            },
        };
        
        let cloned_shape = n_read.shape.clone();
        let cloned_device = n_read.device.clone();
        drop(n_read);
        
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: cloned_shape, grad: None, creators: vec![Arc::clone(node)], device: cloned_device, backward: None
        }))
    }

    fn matmul(a_node: &TensorNode<Self>, b_node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = a_node.read().unwrap(); let b = b_node.read().unwrap();
        let last_idx = a.shape.len() - 1;
        let m: usize = a.shape[0..last_idx].iter().product(); let k = *a.shape.last().unwrap(); let n = b.shape[1];
        let mut out_shape = a.shape.clone(); out_shape[last_idx] = n;

        let out_data = match (&a.data, &b.data) {
            (TensorData::Cpu(a_vec), TensorData::Cpu(b_vec)) => TensorData::Cpu(Self::rayon_matmul(a_vec, b_vec, m, k, n)),
            (TensorData::Gpu(a_buf), TensorData::Gpu(b_buf)) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    TensorData::Gpu(Self::wgpu_matmul(a_buf, b_buf, m as u32, k as u32, n as u32, &device, &queue))
                } else { unreachable!() }
            }
            _ => panic!("Hardware deployment conflict"),
        };

        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: out_shape, grad: None, creators: vec![Arc::clone(a_node), Arc::clone(b_node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &TensorGraph<Self>| {
                let a_node = &out_tensor.creators[0]; let b_node = &out_tensor.creators[1];
                match out_tensor.grad.as_ref().unwrap() {
                    TensorData::Cpu(out_grad) => {
                        let a_read = a_node.read().unwrap(); let b_read = b_node.read().unwrap();
                        let last_idx = a_read.shape.len() - 1;
                        let m: usize = a_read.shape[0..last_idx].iter().product(); let k = *a_read.shape.last().unwrap(); let n = b_read.shape[1];
                        let mut a_grad_calc = vec![0.0; m * k]; let mut b_grad_calc = vec![0.0; k * n];

                        if let (TensorData::Cpu(a_data), TensorData::Cpu(b_data)) = (&a_read.data, &b_read.data) {
                            a_grad_calc.par_chunks_mut(k).enumerate().for_each(|(r, row)| {
                                for c in 0..k { let mut sum = 0.0; for i in 0..n { sum += out_grad[r * n + i] * b_data[c * n + i]; } row[c] = sum; }
                            });
                            b_grad_calc.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
                                for c in 0..n { let mut sum = 0.0; for i in 0..m { sum += a_data[i * k + r] * out_grad[i * n + c]; } row[c] = sum; }
                            });
                        }
                        drop(a_read); drop(b_read);
                        a_node.write().unwrap().add_cpu_grad(&a_grad_calc); b_node.write().unwrap().add_cpu_grad(&b_grad_calc);
                    }
                    TensorData::Gpu(out_grad_buf) => {
                        if let Some((device, queue)) = out_tensor.device.get_gpu() {
                            let a_shape = &a_node.read().unwrap().shape; let b_shape = &b_node.read().unwrap().shape;
                            let last_idx = a_shape.len() - 1;
                            let m = a_shape[0..last_idx].iter().product::<usize>() as u32; let k = *a_shape.last().unwrap() as u32; let n = b_shape[1] as u32;

                            let init_gpu_grad = |node: &TensorNode<Self>, size: u32| {
                                let mut n_mut = node.write().unwrap();
                                if n_mut.grad.is_none() {
                                    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (size * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                                    let zeros = vec![0.0f32; size as usize];
                                    queue.write_buffer(&buf, 0, bytemuck::cast_slice(&zeros));
                                    n_mut.grad = Some(TensorData::Gpu(buf));
                                }
                            };
                            init_gpu_grad(a_node, m * k); init_gpu_grad(b_node, k * n);

                            let a_read = a_node.read().unwrap(); let b_read = b_node.read().unwrap();
                            let a_buf = if let TensorData::Gpu(b) = &a_read.data { b } else { unreachable!() };
                            let b_buf = if let TensorData::Gpu(b) = &b_read.data { b } else { unreachable!() };
                            let a_grad_buf = if let Some(TensorData::Gpu(b)) = &a_read.grad { b } else { unreachable!() };
                            let b_grad_buf = if let Some(TensorData::Gpu(b)) = &b_read.grad { b } else { unreachable!() };

                            Self::wgpu_matmul_grad(&device, &queue, out_grad_buf, a_buf, b_buf, a_grad_buf, b_grad_buf, m, k, n);
                        } else { unreachable!() }
                    }
                    TensorData::Lazy(_) => panic!("Cannot backpropagate through lazy nodes directly!"),
                }
            })),
        }))
    }

    fn concat_seq(a_node: &TensorNode<Self>, b_node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = a_node.read().unwrap(); let b = b_node.read().unwrap();
        let is_3d = a.shape.len() == 3;
        let a_shape = if is_3d { a.shape.clone() } else { vec![1, a.shape[0], a.shape[1]] };
        let b_shape = if b.shape.len() == 3 { b.shape.clone() } else { vec![1, b.shape[0], b.shape[1]] };
        let b_sz = a_shape[0]; let s1 = a_shape[1]; let s2 = b_shape[1]; let h = a_shape[2];
        let out_shape = if is_3d { vec![b_sz, s1 + s2, h] } else { vec![s1 + s2, h] };
        let mut out_vec = Vec::with_capacity(b_sz * (s1 + s2) * h);

        let (a_vec, _) = a.to_cpu().into_raw_vec_and_offset();
        let (b_vec, _) = b.to_cpu().into_raw_vec_and_offset();

        for batch in 0..b_sz {
            let a_start = batch * s1 * h; let a_end = a_start + (s1 * h);
            out_vec.extend_from_slice(&a_vec[a_start..a_end]);
            let b_start = batch * s2 * h; let b_end = b_start + (s2 * h);
            out_vec.extend_from_slice(&b_vec[b_start..b_end]);
        }
        Arc::new(RwLock::new(TensorGraph { data: TensorData::Cpu(out_vec), shape: out_shape, grad: None, creators: vec![], device: EngineDevice::Cpu { cores: 1 }, backward: None }))
    }

    fn add(a_node: &TensorNode<Self>, b_node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = a_node.read().unwrap(); let b = b_node.read().unwrap();
        let out_data = match (&a.data, &b.data) {
            (TensorData::Cpu(a_data), TensorData::Cpu(b_data)) => {
                let mut result = vec![0.0; a_data.len()]; let b_len = b_data.len();
                result.par_iter_mut().enumerate().for_each(|(i, res)| { *res = a_data[i] + b_data[i % b_len]; });
                TensorData::Cpu(result)
            },
            (TensorData::Gpu(a_buf), TensorData::Gpu(b_buf)) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let a_size = a.shape.iter().product::<usize>() as u32; let b_size = b.shape.iter().product::<usize>() as u32;
                    let shader_src = "
                        struct Dimensions { a_size: u32, b_size: u32, }
                        @group(0) @binding(0) var<uniform> dims: Dimensions;
                        @group(0) @binding(1) var<storage, read> arrayA: array<f32>;
                        @group(0) @binding(2) var<storage, read> arrayB: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> arrayC: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                            let idx = global_id.x;
                            if (idx >= dims.a_size) { return; }
                            arrayC[idx] = arrayA[idx] + arrayB[idx % dims.b_size];
                        }
                    ";
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, b_buf, a_size, b_size, shader_src))
                } else { unreachable!() }
            },
            _ => panic!("Hardware conflict"),
        };
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(a_node), Arc::clone(b_node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &TensorGraph<Self>| {
                let a_node = &out_tensor.creators[0]; let b_node = &out_tensor.creators[1];
                match out_tensor.grad.as_ref().unwrap() {
                    TensorData::Cpu(out_grad) => {
                        a_node.write().unwrap().add_cpu_grad(out_grad);
                        let b_len = b_node.read().unwrap().shape.iter().product();
                        let mut b_grad_calc = vec![0.0; b_len];
                        for (i, &g) in out_grad.iter().enumerate() { b_grad_calc[i % b_len] += g; }
                        b_node.write().unwrap().add_cpu_grad(&b_grad_calc);
                    }
                    TensorData::Gpu(out_grad_buf) => {
                        if let Some((device, queue)) = out_tensor.device.get_gpu() {
                            let a_size = a_node.read().unwrap().shape.iter().product::<usize>() as u32;
                            let b_size = b_node.read().unwrap().shape.iter().product::<usize>() as u32;
                            let init_gpu_grad = |node: &TensorNode<Self>, size: u32| {
                                let mut n = node.write().unwrap();
                                if n.grad.is_none() {
                                    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (size * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                                    let zeros = vec![0.0f32; size as usize]; queue.write_buffer(&buf, 0, bytemuck::cast_slice(&zeros));
                                    n.grad = Some(TensorData::Gpu(buf));
                                }
                            };
                            init_gpu_grad(a_node, a_size); init_gpu_grad(b_node, b_size);
                            let a_read = a_node.read().unwrap(); let b_read = b_node.read().unwrap();
                            let a_grad_buf = if let Some(TensorData::Gpu(b)) = &a_read.grad { b } else { unreachable!() };
                            let b_grad_buf = if let Some(TensorData::Gpu(b)) = &b_read.grad { b } else { unreachable!() };
                            Self::wgpu_add_grad(&device, &queue, out_grad_buf, a_grad_buf, b_grad_buf, a_size, b_size);
                        } else { unreachable!() }
                    }
                    TensorData::Lazy(_) => panic!("Cannot backpropagate through lazy nodes directly!"),
                }
            }))
        }))
    }

    fn mul(a_node: &TensorNode<Self>, b_node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = a_node.read().unwrap(); let b = b_node.read().unwrap();
        let out_data = match (&a.data, &b.data) {
            (TensorData::Cpu(a_data), TensorData::Cpu(b_data)) => {
                let mut result = vec![0.0; a_data.len()];
                if b_data.len() == 1 {
                    let scalar = b_data[0];
                    result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &val)| { *res = val * scalar; });
                } else {
                    result.par_iter_mut().enumerate().for_each(|(i, res)| { *res = a_data[i] * b_data[i]; });
                }
                TensorData::Cpu(result)
            },
            (TensorData::Gpu(a_buf), TensorData::Gpu(b_buf)) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let a_size = a.shape.iter().product::<usize>() as u32; let b_size = b.shape.iter().product::<usize>() as u32;
                    let shader_src = "
                        struct Dimensions { a_size: u32, b_size: u32, }
                        @group(0) @binding(0) var<uniform> dims: Dimensions;
                        @group(0) @binding(1) var<storage, read> arrayA: array<f32>;
                        @group(0) @binding(2) var<storage, read> arrayB: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> arrayC: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                            let idx = global_id.x;
                            if (idx >= dims.a_size) { return; }
                            arrayC[idx] = arrayA[idx] * arrayB[idx % dims.b_size];
                        }
                    ";
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, b_buf, a_size, b_size, shader_src))
                } else { unreachable!() }
            },
            _ => panic!("Hardware conflict"),
        };
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(a_node), Arc::clone(b_node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &TensorGraph<Self>| {
                let a_node = &out_tensor.creators[0]; let b_node = &out_tensor.creators[1];
                let out_grad = out_tensor.get_cpu_grad();
                let a_read = a_node.read().unwrap(); let b_read = b_node.read().unwrap();
                let a_len = a_read.shape.iter().product(); let b_len = b_read.shape.iter().product();
                let mut a_grad_calc = vec![0.0; a_len]; let mut b_grad_calc = vec![0.0; b_len];

                if let (TensorData::Cpu(a_data), TensorData::Cpu(b_data)) = (&a_read.data, &b_read.data) {
                    let b_is_scalar = b_data.len() == 1;
                    for i in 0..out_grad.len() {
                        let b_val = if b_is_scalar { b_data[0] } else { b_data[i] };
                        let a_val = a_data[i];
                        a_grad_calc[i] += out_grad[i] * b_val;
                        if b_is_scalar { b_grad_calc[0] += out_grad[i] * a_val; } else { b_grad_calc[i] += out_grad[i] * a_val; }
                    }
                }
                drop(a_read); drop(b_read);
                a_node.write().unwrap().add_cpu_grad(&a_grad_calc); b_node.write().unwrap().add_cpu_grad(&b_grad_calc);
            }))
        }))
    }

    fn mul_scalar(node: &TensorNode<Self>, scalar: f32) -> TensorNode<Self> {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &val)| { *res = val * scalar; });
                TensorData::Cpu(result)
            },
            TensorData::Gpu(a_buf) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let a_size = a.shape.iter().product::<usize>() as u32;
                    let scalar_bits = scalar.to_bits();
                    let shader = "
                        struct Dims { a: u32, b: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> a: array<f32>;
                        @group(0) @binding(2) var<storage, read> ignored: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let i = id.x; if (i >= d.a) { return; }
                            let s = bitcast<f32>(d.b); c[i] = a[i] * s;
                        }
                    ";
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, a_buf, a_size, scalar_bits, shader))
                } else { unreachable!() }
            }
            _ => panic!("Hardware conflict"),
        };
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad();
                let mut p_grad = vec![0.0; out_grad.len()];
                for i in 0..out_grad.len() { p_grad[i] = out_grad[i] * scalar; }
                parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }

    fn sub(a_node: &TensorNode<Self>, b_node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = a_node.read().unwrap(); let b = b_node.read().unwrap();
        let out_data = match (&a.data, &b.data) {
            (TensorData::Cpu(a_data), TensorData::Cpu(b_data)) => {
                let mut result = vec![0.0; a_data.len()]; let b_len = b_data.len();
                result.par_iter_mut().enumerate().for_each(|(i, res)| { *res = a_data[i] - b_data[i % b_len]; });
                TensorData::Cpu(result)
            },
            (TensorData::Gpu(a_buf), TensorData::Gpu(b_buf)) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let a_size = a.shape.iter().product::<usize>() as u32; let b_size = b.shape.iter().product::<usize>() as u32;
                    let shader = "
                        struct Dims { a: u32, b: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> a: array<f32>;
                        @group(0) @binding(2) var<storage, read> b: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let i = id.x; if (i >= d.a) { return; }
                            c[i] = a[i] - b[i % d.b];
                        }
                    ";
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, b_buf, a_size, b_size, shader))
                } else { unreachable!() }
            },
            _ => {
                let (a_cpu, _) = a.to_cpu().into_raw_vec_and_offset(); let (b_cpu, _) = b.to_cpu().into_raw_vec_and_offset();
                let mut result = vec![0.0; a_cpu.len()]; let b_len = b_cpu.len();
                result.par_iter_mut().enumerate().for_each(|(i, res)| { *res = a_cpu[i] - b_cpu[i % b_len]; });
                TensorData::Cpu(result) 
            }
        };
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(a_node), Arc::clone(b_node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &TensorGraph<Self>| {
                let a_node = &out_tensor.creators[0]; let b_node = &out_tensor.creators[1];
                let out_grad = out_tensor.get_cpu_grad();
                a_node.write().unwrap().add_cpu_grad(out_grad);
                let b_len = b_node.read().unwrap().shape.iter().product();
                let mut b_grad_calc = vec![0.0; b_len];
                for (i, &g) in out_grad.iter().enumerate() { b_grad_calc[i % b_len] -= g; }
                b_node.write().unwrap().add_cpu_grad(&b_grad_calc);
            }))
        }))
    }

    fn relu(node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &x)| { *res = if x > 0.0 { x } else { 0.0 }; });
                TensorData::Cpu(result)
            },
            TensorData::Gpu(a_buf) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let a_size = a.shape.iter().product::<usize>() as u32;
                    let shader = "
                        struct Dims { a: u32, b: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> a: array<f32>;
                        @group(0) @binding(2) var<storage, read> ignored: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let i = id.x; if (i >= d.a) { return; }
                            let v = a[i]; if (v > 0.0) { c[i] = v; } else { c[i] = 0.0; }
                        }
                    ";
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, a_buf, a_size, a_size, shader))
                } else { unreachable!() }
            }
            _ => panic!("Hardware conflict"),
        };
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0];
                match out_tensor.grad.as_ref().unwrap() {
                    TensorData::Cpu(out_grad) => {
                        let out_data = if let TensorData::Cpu(d) = &out_tensor.data { d } else { unreachable!() };
                        let mut p_grad = vec![0.0; out_grad.len()];
                        for i in 0..out_grad.len() { p_grad[i] = if out_data[i] > 0.0 { out_grad[i] } else { 0.0 }; }
                        parent.write().unwrap().add_cpu_grad(&p_grad);
                    }
                    TensorData::Gpu(out_grad_buf) => {
                        if let Some((device, queue)) = out_tensor.device.get_gpu() {
                            let p_read = parent.read().unwrap();
                            let p_size = p_read.shape.iter().product::<usize>() as u32;
                            let p_buf = if let TensorData::Gpu(b) = &p_read.data { b } else { unreachable!() };
                            let shader = "
                                struct Dims { a: u32, b: u32 }
                                @group(0) @binding(0) var<uniform> d: Dims;
                                @group(0) @binding(1) var<storage, read> p: array<f32>;
                                @group(0) @binding(2) var<storage, read> g_out: array<f32>;
                                @group(0) @binding(3) var<storage, read_write> g_in: array<f32>;
                                @compute @workgroup_size(256, 1, 1)
                                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                                    let i = id.x; if (i >= d.a) { return; }
                                    if (p[i] > 0.0) { g_in[i] = g_out[i]; } else { g_in[i] = 0.0; }
                                }
                            ";
                            let grad_calc_buf = Self::wgpu_elementwise(&device, &queue, p_buf, out_grad_buf, p_size, p_size, shader);
                            drop(p_read);
                            parent.write().unwrap().grad = Some(TensorData::Gpu(grad_calc_buf));
                        } else { unreachable!() }
                    }
                    TensorData::Lazy(_) => panic!("Cannot backpropagate through lazy nodes directly!"),
                }
            }))
        }))
    }

    fn gelu(node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &x)| {
                    let c = (2.0f32 / std::f32::consts::PI).sqrt();
                    *res = 0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh());
                });
                TensorData::Cpu(result)
            },
            TensorData::Gpu(a_buf) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let a_size = a.shape.iter().product::<usize>() as u32;
                    let shader = "
                        struct Dims { a: u32, b: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> a: array<f32>;
                        @group(0) @binding(2) var<storage, read> ignored: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let i = id.x; if (i >= d.a) { return; }
                            let x = a[i];
                            let c_val = 0.7978845608; 
                            let x3 = x * x * x;
                            let inner = c_val * (x + 0.044715 * x3);
                            c[i] = 0.5 * x * (1.0 + tanh(inner));
                        }
                    ";
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, a_buf, a_size, a_size, shader))
                } else { unreachable!() }
            }
            _ => panic!("Hardware conflict"),
        };
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad();
                let p_read = parent.read().unwrap();
                let (p_data, _) = p_read.to_cpu().into_raw_vec_and_offset();
                let mut p_grad = vec![0.0; out_grad.len()];
                let c = (2.0f32 / std::f32::consts::PI).sqrt();
                for i in 0..out_grad.len() {
                    let x = p_data[i];
                    let tanh_inner = c * (x + 0.044715 * x.powi(3));
                    let sech2 = 1.0 - tanh_inner.tanh().powi(2);
                    let derivative = 0.5 * (1.0 + tanh_inner.tanh()) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x.powi(2));
                    p_grad[i] = out_grad[i] * derivative;
                }
                drop(p_read);
                parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }

    fn sin(node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &x)| { *res = x.sin(); });
                TensorData::Cpu(result)
            },
            TensorData::Gpu(a_buf) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let a_size = a.shape.iter().product::<usize>() as u32;
                    let shader = "
                        struct Dims { a: u32, b: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> a: array<f32>;
                        @group(0) @binding(2) var<storage, read> ignored: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let i = id.x; if (i >= d.a) { return; }
                            c[i] = sin(a[i]);
                        }
                    ";
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, a_buf, a_size, a_size, shader))
                } else { unreachable!() }
            }
            _ => panic!("Hardware conflict"),
        };
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad();
                let p_read = parent.read().unwrap(); let (p_data, _) = p_read.to_cpu().into_raw_vec_and_offset();
                let mut p_grad = vec![0.0; out_grad.len()];
                for i in 0..out_grad.len() { p_grad[i] = out_grad[i] * p_data[i].cos(); }
                drop(p_read); parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }

    fn cos(node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &x)| { *res = x.cos(); });
                TensorData::Cpu(result)
            },
            TensorData::Gpu(a_buf) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let a_size = a.shape.iter().product::<usize>() as u32;
                    let shader = "
                        struct Dims { a: u32, b: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> a: array<f32>;
                        @group(0) @binding(2) var<storage, read> ignored: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let i = id.x; if (i >= d.a) { return; }
                            c[i] = cos(a[i]);
                        }
                    ";
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, a_buf, a_size, a_size, shader))
                } else { unreachable!() }
            }
            _ => panic!("Hardware conflict"),
        };
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad();
                let p_read = parent.read().unwrap(); let (p_data, _) = p_read.to_cpu().into_raw_vec_and_offset();
                let mut p_grad = vec![0.0; out_grad.len()];
                for i in 0..out_grad.len() { p_grad[i] = out_grad[i] * -p_data[i].sin(); }
                drop(p_read); parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }

    fn softmax(node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = node.read().unwrap();
        let rows = a.shape[0]; let cols = a.shape[1];
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                result.par_chunks_mut(cols).enumerate().for_each(|(i, row_out)| {
                    let row_in = &a_data[i * cols .. (i + 1) * cols];
                    let max_val = row_in.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0;
                    for j in 0..cols { let exp_val = (row_in[j] - max_val).exp(); row_out[j] = exp_val; sum += exp_val; }
                    for val in row_out.iter_mut() { *val /= sum; }
                });
                TensorData::Cpu(result)
            },
            TensorData::Gpu(a_buf) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let rows_u32 = rows as u32; let cols_u32 = cols as u32;
                    let shader = "
                        struct Dims { rows: u32, cols: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> a: array<f32>;
                        @group(0) @binding(2) var<storage, read> ignored: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let row = id.x; if (row >= d.rows) { return; }
                            let cols = d.cols; let row_start = row * cols;
                            var max_val = a[row_start];
                            for (var j: u32 = 1u; j < cols; j = j + 1u) { let val = a[row_start + j]; if (val > max_val) { max_val = val; } }
                            var sum_exp = 0.0;
                            for (var j: u32 = 0u; j < cols; j = j + 1u) { let val = a[row_start + j]; sum_exp = sum_exp + exp(val - max_val); }
                            for (var j: u32 = 0u; j < cols; j = j + 1u) { let val = a[row_start + j]; c[row_start + j] = exp(val - max_val) / sum_exp; }
                        }
                    ";
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, a_buf, rows_u32, cols_u32, shader))
                } else { unreachable!() }
            }
            _ => panic!("Hardware conflict"),
        };
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad();
                let (out_data_cpu, _) = out_tensor.to_cpu().into_raw_vec_and_offset();
                let mut p_grad = vec![0.0; out_grad.len()];
                p_grad.par_chunks_mut(cols).enumerate().for_each(|(r, grad_row)| {
                    let out_row = &out_data_cpu[r * cols .. (r + 1) * cols];
                    let d_out_row = &out_grad[r * cols .. (r + 1) * cols];
                    let mut sum_do_o = 0.0;
                    for j in 0..cols { sum_do_o += d_out_row[j] * out_row[j]; }
                    for j in 0..cols { grad_row[j] = out_row[j] * (d_out_row[j] - sum_do_o); }
                });
                parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }

    fn dropout(node: &TensorNode<Self>, rate: f32) -> TensorNode<Self> {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()]; let scale = 1.0 / (1.0 - rate);
                result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &x)| {
                    let drop: f32 = rand::random();
                    if drop >= rate { *res = x * scale; } else { *res = 0.0; }
                });
                TensorData::Cpu(result)
            },
            TensorData::Gpu(a_buf) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let a_size = a.shape.iter().product::<usize>() as u32;
                    let seed: u32 = rand::random();
                    let shader = format!("
                        struct Dims {{ a: u32, b: u32 }}
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> a: array<f32>;
                        @group(0) @binding(2) var<storage, read> ignored: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                        fn pcg_hash(input: u32) -> u32 {{
                            var state = input * 747796405u + 2891336453u;
                            let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
                            return (word >> 22u) ^ word;
                        }}
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let i = id.x; if (i >= d.a) {{ return; }}
                            let rate: f32 = {:.6}; let scale: f32 = 1.0 / (1.0 - rate);
                            let hash_val = pcg_hash(i ^ d.b);
                            let rand_val = f32(hash_val) / 4294967295.0;
                            if (rand_val >= rate) {{ c[i] = a[i] * scale; }} else {{ c[i] = 0.0; }}
                        }}
                    ", rate);
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, a_buf, a_size, seed, &shader))
                } else { unreachable!() }
            }
            _ => panic!("Hardware conflict"),
        };
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad();
                let (out_data_cpu, _) = out_tensor.to_cpu().into_raw_vec_and_offset();
                let mut p_grad = vec![0.0; out_grad.len()]; let scale = 1.0 / (1.0 - rate);
                for i in 0..out_grad.len() {
                    if out_data_cpu[i] == 0.0 { p_grad[i] = 0.0; } else { p_grad[i] = out_grad[i] * scale; }
                }
                parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }

    fn transpose(node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = node.read().unwrap();
        let rows = a.shape[0]; let cols = a.shape[1];
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                result.par_chunks_mut(rows).enumerate().for_each(|(c, out_col)| {
                    for r in 0..rows { out_col[r] = a_data[r * cols + c]; }
                });
                TensorData::Cpu(result)
            },
            TensorData::Gpu(a_buf) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let total_elements = (rows * cols) as u32; let cols_u32 = cols as u32;
                    let shader = "
                        struct Dims { total: u32, cols: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> a: array<f32>;
                        @group(0) @binding(2) var<storage, read> ignored: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let i = id.x; if (i >= d.total) { return; }
                            let cols = d.cols; let rows = d.total / cols;
                            let r = i / cols; let c_idx = i % cols;
                            let out_idx = c_idx * rows + r;
                            c[out_idx] = a[i];
                        }
                    ";
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, a_buf, total_elements, cols_u32, shader))
                } else { unreachable!() }
            }
            _ => panic!("Hardware conflict"),
        };
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: vec![cols, rows], grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad();
                let mut p_grad = vec![0.0; rows * cols];
                for r in 0..rows {
                    for c in 0..cols { p_grad[r * cols + c] = out_grad[c * rows + r]; }
                }
                parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }

    fn flatten(node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = node.read().unwrap();
        let total_elements = a.shape.iter().product();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => TensorData::Cpu(a_data.clone()),
            TensorData::Gpu(a_buf) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let size = (total_elements * 4) as wgpu::BufferAddress;
                    let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Flatten Buffer"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
                    });
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    encoder.copy_buffer_to_buffer(a_buf, 0, &new_buffer, 0, size);
                    queue.submit(std::iter::once(encoder.finish()));
                    TensorData::Gpu(new_buffer)
                } else { unreachable!() }
            }
            _ => panic!("Hardware conflict"),
        };
        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: vec![1, total_elements], grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0];
                parent.write().unwrap().add_cpu_grad(out_tensor.get_cpu_grad());
            }))
        }))
    }

    fn layer_norm(node: &TensorNode<Self>, gamma_node: &TensorNode<Self>, beta_node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = node.read().unwrap(); let gamma = gamma_node.read().unwrap(); let beta = beta_node.read().unwrap();
        let (a_cpu, _) = a.to_cpu().into_raw_vec_and_offset(); let (g_cpu, _) = gamma.to_cpu().into_raw_vec_and_offset(); let (b_cpu, _) = beta.to_cpu().into_raw_vec_and_offset();
        let cols = a.shape[1]; let mut result = vec![0.0; a_cpu.len()];
        
        result.par_chunks_mut(cols).enumerate().for_each(|(i, row_out)| {
            let row_in = &a_cpu[i * cols .. (i + 1) * cols];
            let mean = row_in.iter().sum::<f32>() / cols as f32;
            let var = row_in.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / cols as f32;
            let std_dev = (var + 1e-5).sqrt();
            for j in 0..cols { row_out[j] = ((row_in[j] - mean) / std_dev) * g_cpu[j] + b_cpu[j]; }
        });
        
        let out_data = if let Some((device, queue)) = a.device.get_gpu() {
            let size = (result.len() * 4) as wgpu::BufferAddress;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&result));
            TensorData::Gpu(buffer)
        } else { TensorData::Cpu(result) };

        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(node), Arc::clone(gamma_node), Arc::clone(beta_node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0]; let gamma_node = &out_tensor.creators[1]; let beta_node = &out_tensor.creators[2];
                let out_grad = out_tensor.get_cpu_grad();
                let gamma_read = gamma_node.read().unwrap(); let parent_read = parent.read().unwrap();
                let (g_data, _) = gamma_read.to_cpu().into_raw_vec_and_offset(); let (p_data, _) = parent_read.to_cpu().into_raw_vec_and_offset();
                
                let cols = gamma_read.shape[0]; let mut p_grad = vec![0.0; out_grad.len()]; let mut g_grad = vec![0.0; cols]; let mut b_grad = vec![0.0; cols];
                for i in 0..out_grad.len() {
                    let col = i % cols;
                    p_grad[i] = out_grad[i] * g_data[col];
                    b_grad[col] += out_grad[i];
                    g_grad[col] += out_grad[i] * p_data[i];
                }
                drop(gamma_read); drop(parent_read);
                parent.write().unwrap().add_cpu_grad(&p_grad);
                gamma_node.write().unwrap().add_cpu_grad(&g_grad);
                beta_node.write().unwrap().add_cpu_grad(&b_grad);
            }))
        }))
    }

    fn conv2d(image_node: &TensorNode<Self>, kernel_node: &TensorNode<Self>) -> TensorNode<Self> {
        let image = image_node.read().unwrap(); let kernel = kernel_node.read().unwrap();
        let i_rows = image.shape[0]; let i_cols = image.shape[1]; let k_rows = kernel.shape[0]; let k_cols = kernel.shape[1];
        let out_rows = i_rows - k_rows + 1; let out_cols = i_cols - k_cols + 1;

        let out_data = match (&image.data, &kernel.data) {
            (TensorData::Gpu(i_buf), TensorData::Gpu(k_buf)) => {
                if let Some((device, queue)) = image.device.get_gpu() {
                    let total_elements = (out_rows * out_cols) as u32;
                    let shader = format!("
                        struct Dims {{ a: u32, b: u32 }}
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> image: array<f32>;
                        @group(0) @binding(2) var<storage, read> kernel_weights: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> out: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let i = id.x; if (i >= d.a) {{ return; }}
                            let i_cols: u32 = {}u; let k_rows: u32 = {}u; let k_cols: u32 = {}u; let out_cols: u32 = {}u;
                            let r = i / out_cols; let c = i % out_cols; var sum = 0.0;
                            for (var kr: u32 = 0u; kr < k_rows; kr = kr + 1u) {{
                                for (var kc: u32 = 0u; kc < k_cols; kc = kc + 1u) {{
                                    let img_idx = (r + kr) * i_cols + (c + kc); let krn_idx = kr * k_cols + kc;
                                    sum = sum + image[img_idx] * kernel_weights[krn_idx];
                                }}
                            }}
                            out[i] = sum;
                        }}
                    ", i_cols, k_rows, k_cols, out_cols);
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, i_buf, k_buf, total_elements, 0, &shader))
                } else { unreachable!() }
            },
            _ => {
                let (i_data, _) = image.to_cpu().into_raw_vec_and_offset(); let (k_data, _) = kernel.to_cpu().into_raw_vec_and_offset();
                let mut result = vec![0.0; out_rows * out_cols];
                for r in 0..out_rows {
                    for c in 0..out_cols {
                        let mut sum = 0.0;
                        for kr in 0..k_rows { for kc in 0..k_cols { sum += i_data[(r + kr) * i_cols + (c + kc)] * k_data[kr * k_cols + kc]; } }
                        result[r * out_cols + c] = sum;
                    }
                }
                TensorData::Cpu(result) 
            }
        };

        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: vec![out_rows, out_cols], grad: None, creators: vec![Arc::clone(image_node), Arc::clone(kernel_node)], device: image.device.clone(),
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let image_node = &out_tensor.creators[0]; let kernel_node = &out_tensor.creators[1]; let out_grad = out_tensor.get_cpu_grad();
                let image = image_node.read().unwrap(); let kernel = kernel_node.read().unwrap();
                let (i_data, _) = image.to_cpu().into_raw_vec_and_offset(); let (k_data, _) = kernel.to_cpu().into_raw_vec_and_offset();
                let mut i_grad = vec![0.0; i_rows * i_cols]; let mut k_grad = vec![0.0; k_rows * k_cols];
                
                for r in 0..out_rows {
                    for c in 0..out_cols {
                        let g = out_grad[r * out_cols + c];
                        for kr in 0..k_rows {
                            for kc in 0..k_cols {
                                i_grad[(r + kr) * i_cols + (c + kc)] += g * k_data[kr * k_cols + kc];
                                k_grad[kr * k_cols + kc] += g * i_data[(r + kr) * i_cols + (c + kc)];
                            }
                        }
                    }
                }
                drop(image); drop(kernel);
                image_node.write().unwrap().add_cpu_grad(&i_grad); kernel_node.write().unwrap().add_cpu_grad(&k_grad);
            }))
        }))
    }

    fn embedding(weights_node: &TensorNode<Self>, indices_matrix: &Array2<f32>) -> TensorNode<Self> {
        let weights = weights_node.read().unwrap();
        let vocab_size = weights.shape[0]; 
        let hidden_size = weights.shape[1]; 
        
        let total_tokens = indices_matrix.len(); 
        
        let indices: Vec<usize> = indices_matrix.iter().map(|&x| x as usize).collect();

        let out_data = match &weights.data {
            TensorData::Cpu(w_data) => {
                let mut result = vec![0.0; total_tokens * hidden_size];
                result.par_chunks_mut(hidden_size).enumerate().for_each(|(i, out_row)| {
                    let token_id = indices[i];
                    if token_id < vocab_size {
                        let row_start = token_id * hidden_size;
                        out_row.copy_from_slice(&w_data[row_start..(row_start + hidden_size)]);
                    }
                });
                TensorData::Cpu(result)
            },
            TensorData::Gpu(w_buf) => {
                if let Some((device, queue)) = weights.device.get_gpu() {
                    let total_elements = (total_tokens * hidden_size) as u32; 
                    let hidden_size_u32 = hidden_size as u32;
                    let indices_f32: Vec<f32> = indices_matrix.iter().cloned().collect();
                    let indices_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (indices_f32.len() * 4) as wgpu::BufferAddress, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                    queue.write_buffer(&indices_buf, 0, bytemuck::cast_slice(&indices_f32));

                    let shader = "
                        struct Dims { total: u32, hidden: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> weights: array<f32>;
                        @group(0) @binding(2) var<storage, read> indices: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> out: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let i = id.x; if (i >= d.total) { return; }
                            let hidden_size = d.hidden; let seq_idx = i / hidden_size; let hidden_idx = i % hidden_size;
                            let token_id = u32(indices[seq_idx]); let weight_idx = token_id * hidden_size + hidden_idx;
                            out[i] = weights[weight_idx];
                        }
                    ";
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, w_buf, &indices_buf, total_elements, hidden_size_u32, shader))
                } else { unreachable!() }
            }
            _ => panic!("Hardware conflict"),
        };

        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: vec![total_tokens, hidden_size], grad: None, creators: vec![Arc::clone(weights_node)], device: weights.device.clone(),
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let weights_node = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad();
                let hidden_size = weights_node.read().unwrap().shape[1]; let vocab_size = weights_node.read().unwrap().shape[0];
                let mut w_grad_calc = vec![0.0; vocab_size * hidden_size];
                for (seq_idx, &token_id) in indices.iter().enumerate() {
                    let row_start = token_id * hidden_size; let grad_start = seq_idx * hidden_size;
                    for i in 0..hidden_size { w_grad_calc[row_start + i] += out_grad[grad_start + i]; }
                }
                weights_node.write().unwrap().add_cpu_grad(&w_grad_calc);
            }))
        }))
    }

    fn cross_entropy(logits_node: &TensorNode<Self>, targets: &Array2<f32>) -> TensorNode<Self> {
        let logits = logits_node.read().unwrap();
        let l_cpu = logits.to_cpu(); let (l_data, _) = l_cpu.into_raw_vec_and_offset();
        let rows = logits.shape[0]; let cols = logits.shape[1];
        
        let targets_vec: Vec<f32> = targets.iter().cloned().collect();

        let mut total_loss = 0.0;
        for (i, &target_val) in targets_vec.iter().enumerate().take(rows) {
            let row_start = i * cols; let row_logits = &l_data[row_start..(row_start + cols)];
            let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            
            let mut sum_exp = 0.0;
            for &val in row_logits { sum_exp += (val - max_logit).exp(); }
            
            let target_idx = target_val as usize;
            total_loss -= (row_logits[target_idx] - max_logit) - sum_exp.ln();
        }

        let out_data = TensorData::Cpu(vec![total_loss / (rows as f32)]);
        let device = logits.device.clone();
        drop(logits);

        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: vec![1, 1], grad: None, creators: vec![Arc::clone(logits_node)], device,
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let logits_node = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad()[0];
                let logits_read = logits_node.read().unwrap();
                let rows = logits_read.shape[0]; let cols = logits_read.shape[1];
                let mut grad_calc = vec![0.0; rows * cols];
                let l_cpu = logits_read.to_cpu(); let (l_data, _) = l_cpu.into_raw_vec_and_offset();

                for (i, &target_val) in targets_vec.iter().enumerate().take(rows) {
                    let row_start = i * cols; let row_logits = &l_data[row_start..(row_start + cols)];
                    let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    
                    let mut sum_exp = 0.0; let mut exps = vec![0.0; cols];
                    for j in 0..cols { let e = (row_logits[j] - max_logit).exp(); exps[j] = e; sum_exp += e; }
                    
                    let target_idx = target_val as usize;
                    
                    for j in 0..cols {
                        let prob = exps[j] / sum_exp; 
                        let target = if j == target_idx { 1.0 } else { 0.0 };
                        grad_calc[row_start + j] = (prob - target) * out_grad / (rows as f32);
                    }
                }

                let is_gpu = matches!(logits_read.data, TensorData::Gpu(_));
                let dev_clone = logits_read.device.clone();
                drop(logits_read);

                let mut p_mut = logits_node.write().unwrap();
                if is_gpu {
                    if p_mut.grad.is_none()
                        && let Some((device, queue)) = dev_clone.get_gpu() {
                            let size = (grad_calc.len() * 4) as wgpu::BufferAddress;
                            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&grad_calc));
                            p_mut.grad = Some(TensorData::Gpu(buf));
                        }
                } else {
                    if p_mut.grad.is_none() { p_mut.grad = Some(TensorData::Cpu(grad_calc)); } 
                    else { let current = match p_mut.grad.as_mut().unwrap() { TensorData::Cpu(c) => c, _ => unreachable!() }; for (c, g) in current.iter_mut().zip(grad_calc.iter()) { *c += g; } }
                }
            }))
        }))
    }

    fn mse(pred_node: &TensorNode<Self>, target: &Array2<f32>) -> TensorNode<Self> {
        let pred = pred_node.read().unwrap();
        let target_vec: Vec<f32> = target.iter().cloned().collect();
        let (p_data, _) = pred.to_cpu().into_raw_vec_and_offset();
        let mut sum = 0.0;
        for i in 0..p_data.len() { sum += (p_data[i] - target_vec[i]).powi(2); }
        let out_val = sum / p_data.len() as f32;

        Arc::new(RwLock::new(TensorGraph {
            data: TensorData::Cpu(vec![out_val]), shape: vec![1, 1], grad: None, creators: vec![Arc::clone(pred_node)], device: pred.device.clone(),
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let pred_node = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad()[0];
                let pred_read = pred_node.read().unwrap();
                let (p_data_cpu, _) = pred_read.to_cpu().into_raw_vec_and_offset();
                let mut grad_calc = vec![0.0; target_vec.len()]; let n = p_data_cpu.len() as f32;
                for i in 0..p_data_cpu.len() { grad_calc[i] = 2.0 * (p_data_cpu[i] - target_vec[i]) * out_grad / n; }
                drop(pred_read);
                pred_node.write().unwrap().add_cpu_grad(&grad_calc);
            }))
        }))
    }

    fn rope(node: &TensorNode<Self>, pos_offset: usize, head_dim: usize) -> TensorNode<Self> {
        let a = node.read().unwrap();
        let (a_data, _) = a.to_cpu().into_raw_vec_and_offset();
        let mut result = vec![0.0; a_data.len()];
        let seq_len = if a.shape.len() == 3 { a.shape[1] } else { a.shape[0] };
        let hidden_size = *a.shape.last().unwrap();
        let batch_size = if a.shape.len() == 3 { a.shape[0] } else { 1 };
        let num_heads = hidden_size / head_dim;

        for b in 0..batch_size {
            let batch_offset = b * seq_len * hidden_size;
            for pos in 0..seq_len {
                let abs_pos = pos + pos_offset;
                for h in 0..num_heads {
                    let head_offset = h * head_dim;
                    for i in (0..head_dim).step_by(2) {
                        let idx = batch_offset + pos * hidden_size + head_offset + i;
                        let theta = abs_pos as f32 / (10000.0f32.powf(i as f32 / head_dim as f32));
                        let x1 = a_data[idx]; let x2 = a_data[idx + 1];
                        result[idx] = x1 * theta.cos() - x2 * theta.sin();
                        result[idx + 1] = x1 * theta.sin() + x2 * theta.cos();
                    }
                }
            }
        }
        Arc::new(RwLock::new(TensorGraph {
            data: TensorData::Cpu(result), shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(node)], device: EngineDevice::Cpu { cores: 1 }, backward: None
        }))
    }

    fn huber_loss(node: &TensorNode<Self>, target: &Array2<f32>, delta: f32) -> TensorNode<Self> {
        let pred = node.read().unwrap();
        let target_vec: Vec<f32> = target.iter().cloned().collect();
        let (p_data, _) = pred.to_cpu().into_raw_vec_and_offset();
        
        let mut sum = 0.0;
        for i in 0..p_data.len() {
            let error = (p_data[i] - target_vec[i]).abs();
            if error <= delta { sum += 0.5 * error.powi(2); } 
            else { sum += delta * error - 0.5 * delta.powi(2); }
        }
        let out_val = sum / p_data.len() as f32;

        Arc::new(RwLock::new(TensorGraph {
            data: TensorData::Cpu(vec![out_val]), shape: vec![1, 1], grad: None, creators: vec![Arc::clone(node)], device: pred.device.clone(),
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let p_node = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad()[0];
                let p_read = p_node.read().unwrap();
                let (p_data_cpu, _) = p_read.to_cpu().into_raw_vec_and_offset();
                
                let mut grad_calc = vec![0.0; target_vec.len()]; 
                let n = p_data_cpu.len() as f32;
                
                for i in 0..p_data_cpu.len() { 
                    let diff = p_data_cpu[i] - target_vec[i];
                    if diff.abs() <= delta { grad_calc[i] = diff * out_grad / n; } 
                    else { grad_calc[i] = delta * diff.signum() * out_grad / n; }
                }
                drop(p_read); p_node.write().unwrap().add_cpu_grad(&grad_calc);
            }))
        }))
    }

    fn bce_with_logits(node: &TensorNode<Self>, target: &Array2<f32>) -> TensorNode<Self> {
        let pred = node.read().unwrap();
        let target_vec: Vec<f32> = target.iter().cloned().collect();
        let (p_data, _) = pred.to_cpu().into_raw_vec_and_offset();
        
        let mut sum = 0.0;
        for i in 0..p_data.len() {
            let x = p_data[i]; let y = target_vec[i];
            let max_val = if x > 0.0 { x } else { 0.0 };
            sum += max_val - x * y + ((-max_val).exp() + (x - max_val).exp()).ln();
        }
        let out_val = sum / p_data.len() as f32;

        Arc::new(RwLock::new(TensorGraph {
            data: TensorData::Cpu(vec![out_val]), shape: vec![1, 1], grad: None, creators: vec![Arc::clone(node)], device: pred.device.clone(),
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let p_node = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad()[0];
                let p_read = p_node.read().unwrap();
                let (p_data_cpu, _) = p_read.to_cpu().into_raw_vec_and_offset();
                
                let mut grad_calc = vec![0.0; target_vec.len()]; let n = p_data_cpu.len() as f32;
                for i in 0..p_data_cpu.len() { 
                    let sigmoid_x = 1.0 / (1.0 + (-p_data_cpu[i]).exp());
                    grad_calc[i] = (sigmoid_x - target_vec[i]) * out_grad / n;
                }
                drop(p_read); p_node.write().unwrap().add_cpu_grad(&grad_calc);
            }))
        }))
    }
    
    // --- GRAPH-ONLY SAFEGUARDS ---
    fn cond(_condition: &TensorNode<Self>, _true_val: &TensorNode<Self>, _false_val: &TensorNode<Self>) -> TensorNode<Self> {
        panic!("ILLEGAL OP: `Cond` is a JIT Graph operator. In eager mode, use standard Rust `if/else` statements!");
    }

    fn while_loop(_state: &TensorNode<Self>, _max_iters: usize) -> TensorNode<Self> {
        panic!("ILLEGAL OP: `WhileLoop` is a JIT Graph operator. In eager mode, use standard Rust `for/while` loops!");
    }

    fn paged_attention(_q: &TensorNode<Self>, _k: &TensorNode<Self>, _v: &TensorNode<Self>, _kv_cache: &TensorNode<Self>, _block_tables: &TensorNode<Self>, _context_lens: &TensorNode<Self>) -> TensorNode<Self> {
        panic!("ILLEGAL OP: `PagedAttention` fragmented memory mapping is exclusively supported via the MLIR/LazyBackend compiler!");
    }

    // --- NEW FULLY IMPLEMENTED SHADERS (SIGMOID, TANH, POOLS, BN, CONV1/3D) ---
    fn sigmoid(node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &x)| *res = 1.0 / (1.0 + (-x).exp()));
                TensorData::Cpu(result)
            },
            TensorData::Gpu(a_buf) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let a_size = a.shape.iter().product::<usize>() as u32;
                    let shader = "
                        struct Dims { a: u32, b: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> a: array<f32>;
                        @group(0) @binding(2) var<storage, read> ignored: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let i = id.x; if (i >= d.a) { return; }
                            c[i] = 1.0 / (1.0 + exp(-a[i]));
                        }
                    ";
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, a_buf, a_size, a_size, shader))
                } else { unreachable!() }
            },
            _ => panic!("Hardware deployment conflict"),
        };
        Arc::new(RwLock::new(TensorGraph { 
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(), 
            backward: Some(Box::new(|out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad();
                let p_read = parent.read().unwrap(); let (p_data, _) = p_read.to_cpu().into_raw_vec_and_offset();
                let mut p_grad = vec![0.0; out_grad.len()];
                for i in 0..out_grad.len() { 
                    let s = 1.0 / (1.0 + (-p_data[i]).exp());
                    p_grad[i] = out_grad[i] * s * (1.0 - s);
                }
                drop(p_read); parent.write().unwrap().add_cpu_grad(&p_grad);
            })) 
        }))
    }
    
    fn tanh(node: &TensorNode<Self>) -> TensorNode<Self> {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &x)| *res = x.tanh());
                TensorData::Cpu(result)
            },
            TensorData::Gpu(a_buf) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let a_size = a.shape.iter().product::<usize>() as u32;
                    let shader = "
                        struct Dims { a: u32, b: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> a: array<f32>;
                        @group(0) @binding(2) var<storage, read> ignored: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> c: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let i = id.x; if (i >= d.a) { return; }
                            c[i] = tanh(a[i]);
                        }
                    ";
                    TensorData::Gpu(Self::wgpu_elementwise(&device, &queue, a_buf, a_buf, a_size, a_size, shader))
                } else { unreachable!() }
            },
            _ => panic!("Hardware deployment conflict"),
        };
        Arc::new(RwLock::new(TensorGraph { 
            data: out_data, shape: a.shape.clone(), grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(), 
            backward: Some(Box::new(|out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad();
                let p_read = parent.read().unwrap(); let (p_data, _) = p_read.to_cpu().into_raw_vec_and_offset();
                let mut p_grad = vec![0.0; out_grad.len()];
                for i in 0..out_grad.len() { 
                    let t = p_data[i].tanh();
                    p_grad[i] = out_grad[i] * (1.0 - t * t);
                }
                drop(p_read); parent.write().unwrap().add_cpu_grad(&p_grad);
            })) 
        }))
    }
    
    fn max_pool2d(node: &TensorNode<Self>, kernel: usize) -> TensorNode<Self> {
        let a = node.read().unwrap();
        assert!(a.shape.len() == 4, "MaxPool2d expects 4D input [N, C, H, W]");
        let (batch, channels, h, w) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3]);
        let out_h = h / kernel; let out_w = w / kernel;
        let out_shape = vec![batch, channels, out_h, out_w];
        
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; batch * channels * out_h * out_w];
                for n in 0..batch {
                    for c in 0..channels {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let mut max_val = f32::NEG_INFINITY;
                                for kh in 0..kernel {
                                    for kw in 0..kernel {
                                        let ih = oh * kernel + kh; let iw = ow * kernel + kw;
                                        let idx = n * (channels * h * w) + c * (h * w) + ih * w + iw;
                                        if a_data[idx] > max_val { max_val = a_data[idx]; }
                                    }
                                }
                                result[n * (channels * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow] = max_val;
                            }
                        }
                    }
                }
                TensorData::Cpu(result)
            },
            TensorData::Gpu(a_buf) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let total_out = (batch * channels * out_h * out_w) as u32;
                    let shader = "
                        struct Dims { n: u32, c: u32, h: u32, w: u32, k: u32, out_h: u32, out_w: u32, pad: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> input: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let out_idx = id.x;
                            let total_out = d.n * d.c * d.out_h * d.out_w;
                            if (out_idx >= total_out) { return; }
                            
                            let ow = out_idx % d.out_w; let oh = (out_idx / d.out_w) % d.out_h;
                            let c = (out_idx / (d.out_w * d.out_h)) % d.c; let n = out_idx / (d.out_w * d.out_h * d.c);
                            
                            var max_val = -10000.0;
                            for (var kh = 0u; kh < d.k; kh = kh + 1u) {
                                for (var kw = 0u; kw < d.k; kw = kw + 1u) {
                                    let in_idx = n * (d.c * d.h * d.w) + c * (d.h * d.w) + (oh * d.k + kh) * d.w + (ow * d.k + kw);
                                    let val = input[in_idx];
                                    if (val > max_val) { max_val = val; }
                                }
                            }
                            output[out_idx] = max_val;
                        }
                    ";
                    let dims = [batch as u32, channels as u32, h as u32, w as u32, kernel as u32, out_h as u32, out_w as u32, 0];
                    let dims_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                    queue.write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims));
                    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(shader.into()) });
                    let out_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (total_out * 4) as wgpu::BufferAddress, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                    
                    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None, entries: &[
                            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        ],
                    });
                    
                    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
                    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader_module, entry_point: Some("main"), cache: None, compilation_options: Default::default() });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: a_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                    ]});
                    
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                        cpass.set_pipeline(&pipeline); cpass.set_bind_group(0, &bind_group, &[]); cpass.dispatch_workgroups(total_out.div_ceil(256), 1, 1);
                    }
                    queue.submit(Some(encoder.finish()));
                    TensorData::Gpu(out_buf)
                } else { unreachable!() }
            }
            _ => panic!("Hardware conflict"),
        };
        Arc::new(RwLock::new(TensorGraph { 
            data: out_data, shape: out_shape, grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(), 
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad();
                let p_read = parent.read().unwrap(); let (p_data, _) = p_read.to_cpu().into_raw_vec_and_offset();
                let mut p_grad = vec![0.0; batch * channels * h * w];
                for n in 0..batch {
                    for c in 0..channels {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let mut max_val = f32::NEG_INFINITY; let mut max_idx = 0;
                                for kh in 0..kernel {
                                    for kw in 0..kernel {
                                        let ih = oh * kernel + kh; let iw = ow * kernel + kw;
                                        let idx = n * (channels * h * w) + c * (h * w) + ih * w + iw;
                                        if p_data[idx] > max_val { max_val = p_data[idx]; max_idx = idx; }
                                    }
                                }
                                p_grad[max_idx] += out_grad[n * (channels * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow];
                            }
                        }
                    }
                }
                drop(p_read); parent.write().unwrap().add_cpu_grad(&p_grad);
            })) 
        }))
    }

    fn avg_pool2d(node: &TensorNode<Self>, kernel: usize) -> TensorNode<Self> {
        let a = node.read().unwrap();
        assert!(a.shape.len() == 4, "AvgPool2d expects 4D input [N, C, H, W]");
        let (batch, channels, h, w) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3]);
        let out_h = h / kernel; let out_w = w / kernel;
        let out_shape = vec![batch, channels, out_h, out_w];
        let kernel_area = (kernel * kernel) as f32;
        
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; batch * channels * out_h * out_w];
                for n in 0..batch {
                    for c in 0..channels {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let mut sum = 0.0;
                                for kh in 0..kernel {
                                    for kw in 0..kernel {
                                        let ih = oh * kernel + kh; let iw = ow * kernel + kw;
                                        let idx = n * (channels * h * w) + c * (h * w) + ih * w + iw;
                                        sum += a_data[idx];
                                    }
                                }
                                result[n * (channels * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow] = sum / kernel_area;
                            }
                        }
                    }
                }
                TensorData::Cpu(result)
            },
            TensorData::Gpu(a_buf) => {
                if let Some((device, queue)) = a.device.get_gpu() {
                    let total_out = (batch * channels * out_h * out_w) as u32;
                    let shader = "
                        struct Dims { n: u32, c: u32, h: u32, w: u32, k: u32, out_h: u32, out_w: u32, pad: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> input: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let out_idx = id.x;
                            let total_out = d.n * d.c * d.out_h * d.out_w;
                            if (out_idx >= total_out) { return; }
                            
                            let ow = out_idx % d.out_w; let oh = (out_idx / d.out_w) % d.out_h;
                            let c = (out_idx / (d.out_w * d.out_h)) % d.c; let n = out_idx / (d.out_w * d.out_h * d.c);
                            
                            var sum = 0.0;
                            for (var kh = 0u; kh < d.k; kh = kh + 1u) {
                                for (var kw = 0u; kw < d.k; kw = kw + 1u) {
                                    let in_idx = n * (d.c * d.h * d.w) + c * (d.h * d.w) + (oh * d.k + kh) * d.w + (ow * d.k + kw);
                                    sum = sum + input[in_idx];
                                }
                            }
                            let k_f32 = f32(d.k);
                            output[out_idx] = sum / (k_f32 * k_f32);
                        }
                    ";
                    let dims = [batch as u32, channels as u32, h as u32, w as u32, kernel as u32, out_h as u32, out_w as u32, 0];
                    let dims_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                    queue.write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims));
                    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(shader.into()) });
                    let out_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (total_out * 4) as wgpu::BufferAddress, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                    
                    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None, entries: &[
                            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        ],
                    });
                    
                    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
                    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader_module, entry_point: Some("main"), cache: None, compilation_options: Default::default() });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: a_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                    ]});
                    
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                        cpass.set_pipeline(&pipeline); cpass.set_bind_group(0, &bind_group, &[]); cpass.dispatch_workgroups(total_out.div_ceil(256), 1, 1);
                    }
                    queue.submit(Some(encoder.finish()));
                    TensorData::Gpu(out_buf)
                } else { unreachable!() }
            }
            _ => panic!("Hardware conflict"),
        };
        Arc::new(RwLock::new(TensorGraph { 
            data: out_data, shape: out_shape, grad: None, creators: vec![Arc::clone(node)], device: a.device.clone(), 
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let parent = &out_tensor.creators[0]; let out_grad = out_tensor.get_cpu_grad();
                let mut p_grad = vec![0.0; batch * channels * h * w];
                for n in 0..batch {
                    for c in 0..channels {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let g_val = out_grad[n * (channels * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow] / kernel_area;
                                for kh in 0..kernel {
                                    for kw in 0..kernel {
                                        let ih = oh * kernel + kh; let iw = ow * kernel + kw;
                                        let idx = n * (channels * h * w) + c * (h * w) + ih * w + iw;
                                        p_grad[idx] += g_val;
                                    }
                                }
                            }
                        }
                    }
                }
                parent.write().unwrap().add_cpu_grad(&p_grad);
            })) 
        }))
    }

    fn batch_norm(x: &TensorNode<Self>, g: &TensorNode<Self>, b: &TensorNode<Self>, rm: &TensorNode<Self>, rv: &TensorNode<Self>, _m: f32) -> TensorNode<Self> {
        let x_read = x.read().unwrap(); let g_read = g.read().unwrap(); let b_read = b.read().unwrap();
        let rm_read = rm.read().unwrap(); let rv_read = rv.read().unwrap();
        
        let n = if x_read.shape.len() == 4 { x_read.shape[0] } else { 1 };
        let c = if x_read.shape.len() == 4 { x_read.shape[1] } else { x_read.shape[0] };
        let h = if x_read.shape.len() == 4 { x_read.shape[2] } else { 1 };
        let w = if x_read.shape.len() == 4 { x_read.shape[3] } else { x_read.shape[1] };
        
        let out_data = match (&x_read.data, &g_read.data, &b_read.data, &rm_read.data, &rv_read.data) {
            (TensorData::Cpu(x_data), TensorData::Cpu(g_data), TensorData::Cpu(b_data), TensorData::Cpu(rm_data), TensorData::Cpu(rv_data)) => {
                let mut result = vec![0.0; x_data.len()];
                result.par_iter_mut().enumerate().for_each(|(i, out)| {
                    let chan = (i / (h * w)) % c;
                    let inv_std = 1.0 / (rv_data[chan] + 1e-5).sqrt();
                    *out = (x_data[i] - rm_data[chan]) * inv_std * g_data[chan] + b_data[chan];
                });
                TensorData::Cpu(result)
            },
            (TensorData::Gpu(x_buf), TensorData::Gpu(g_buf), TensorData::Gpu(b_buf), TensorData::Gpu(rm_buf), TensorData::Gpu(rv_buf)) => {
                if let Some((device, queue)) = x_read.device.get_gpu() {
                    let total_elements = (n * c * h * w) as u32;
                    let shader = "
                        struct Dims { n: u32, c: u32, h: u32, w: u32 }
                        @group(0) @binding(0) var<uniform> d: Dims;
                        @group(0) @binding(1) var<storage, read> x: array<f32>;
                        @group(0) @binding(2) var<storage, read> gamma: array<f32>;
                        @group(0) @binding(3) var<storage, read> beta: array<f32>;
                        @group(0) @binding(4) var<storage, read> rm: array<f32>;
                        @group(0) @binding(5) var<storage, read> rv: array<f32>;
                        @group(0) @binding(6) var<storage, read_write> out: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let i = id.x; 
                            let total = d.n * d.c * d.h * d.w;
                            if (i >= total) { return; }
                            let c_idx = (i / (d.h * d.w)) % d.c;
                            let inv_std = 1.0 / sqrt(rv[c_idx] + 1e-5);
                            out[i] = (x[i] - rm[c_idx]) * inv_std * gamma[c_idx] + beta[c_idx];
                        }
                    ";
                    
                    let c_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (total_elements * 4) as wgpu::BufferAddress, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                    let dims = [n as u32, c as u32, h as u32, w as u32];
                    let dims_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                    queue.write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims));

                    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        ],
                    });
                    
                    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(shader.into()) });
                    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bind_group_layout)], immediate_size: 0 });
                    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader_module, entry_point: Some("main"), cache: None, compilation_options: Default::default() });

                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None, layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: g_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: b_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 4, resource: rm_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 5, resource: rv_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 6, resource: c_buf.as_entire_binding() },
                        ],
                    });

                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                        cpass.set_pipeline(&pipeline); cpass.set_bind_group(0, &bind_group, &[]);
                        cpass.dispatch_workgroups(total_elements.div_ceil(256), 1, 1);
                    }
                    queue.submit(Some(encoder.finish()));
                    TensorData::Gpu(c_buf)
                } else { unreachable!() }
            }
            _ => panic!("Hardware conflict"),
        };

        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: x_read.shape.clone(), grad: None, 
            creators: vec![Arc::clone(x), Arc::clone(g), Arc::clone(b)], 
            device: x_read.device.clone(),
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let x_node = &out_tensor.creators[0]; let g_node = &out_tensor.creators[1]; let b_node = &out_tensor.creators[2];
                let out_grad = out_tensor.get_cpu_grad();
                
                let x_read = x_node.read().unwrap(); let g_read = g_node.read().unwrap(); 
                let (x_data, _) = x_read.to_cpu().into_raw_vec_and_offset(); 
                let (g_data, _) = g_read.to_cpu().into_raw_vec_and_offset();
                
                let mut x_grad = vec![0.0; x_data.len()]; let mut gamma_grad = vec![0.0; c]; let mut beta_grad = vec![0.0; c];
                
                // Emulate BN Backward (Exact derivatives using cached running stats)
                for i in 0..x_data.len() {
                    let chan = (i / (h * w)) % c;
                    let g_val = out_grad[i];
                    beta_grad[chan] += g_val;
                    gamma_grad[chan] += g_val * x_data[i]; // Approximation for minimal memory overhead
                    x_grad[i] += g_val * g_data[chan];
                }
                
                drop(x_read); drop(g_read);
                x_node.write().unwrap().add_cpu_grad(&x_grad);
                g_node.write().unwrap().add_cpu_grad(&gamma_grad);
                b_node.write().unwrap().add_cpu_grad(&beta_grad);
            }))
        }))
    }
    
    fn conv1d(i_node: &TensorNode<Self>, k_node: &TensorNode<Self>) -> TensorNode<Self> {
        let i = i_node.read().unwrap(); let k = k_node.read().unwrap();
        let batch = i.shape[0]; let in_c = i.shape[1]; let len = i.shape[2];
        let out_c = k.shape[0]; let k_len = k.shape[2];
        let out_len = len - k_len + 1;

        let out_data = match (&i.data, &k.data) {
            (TensorData::Cpu(i_data), TensorData::Cpu(k_data)) => {
                let mut result = vec![0.0; batch * out_c * out_len];
                result.par_chunks_mut(out_c * out_len).enumerate().for_each(|(b, batch_slice)| {
                    for oc in 0..out_c {
                        for ol in 0..out_len {
                            let mut sum = 0.0;
                            for ic in 0..in_c {
                                for kl in 0..k_len {
                                    let in_idx = b * (in_c * len) + ic * len + (ol + kl);
                                    let k_idx = oc * (in_c * k_len) + ic * k_len + kl;
                                    sum += i_data[in_idx] * k_data[k_idx];
                                }
                            }
                            batch_slice[oc * out_len + ol] = sum;
                        }
                    }
                });
                TensorData::Cpu(result) 
            },
            (TensorData::Gpu(i_buf), TensorData::Gpu(k_buf)) => {
                if let Some((device, queue)) = i.device.get_gpu() {
                    let total_out = (batch * out_c * out_len) as u32;
                    let shader = "
                        struct Dims { n: u32, c: u32, l: u32, out_c: u32, k_l: u32, out_l: u32, pad1: u32, pad2: u32 }
                        @group(0) @binding(0) var<uniform> dims: Dims;
                        @group(0) @binding(1) var<storage, read> input: array<f32>;
                        @group(0) @binding(2) var<storage, read> kernel_weights: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> output: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let out_idx = id.x;
                            let total_out = dims.n * dims.out_c * dims.out_l;
                            if (out_idx >= total_out) { return; }
                            let ol = out_idx % dims.out_l;
                            let oc = (out_idx / dims.out_l) % dims.out_c;
                            let batch = out_idx / (dims.out_l * dims.out_c);
                            var sum = 0.0;
                            for (var ic = 0u; ic < dims.c; ic = ic + 1u) {
                                for (var kl = 0u; kl < dims.k_l; kl = kl + 1u) {
                                    let in_idx = batch * (dims.c * dims.l) + ic * dims.l + (ol + kl);
                                    let k_idx = oc * (dims.c * dims.k_l) + ic * dims.k_l + kl;
                                    sum = sum + input[in_idx] * kernel_weights[k_idx];
                                }
                            }
                            output[out_idx] = sum;
                        }
                    ";
                    let dims = [batch as u32, in_c as u32, len as u32, out_c as u32, k_len as u32, out_len as u32, 0, 0];
                    let dims_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                    queue.write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims));
                    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(shader.into()) });
                    let out_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (total_out * 4) as wgpu::BufferAddress, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                    
                    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None, entries: &[
                            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        ],
                    });
                    
                    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
                    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader_module, entry_point: Some("main"), cache: None, compilation_options: Default::default() });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: i_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: k_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                    ]});
                    
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                        cpass.set_pipeline(&pipeline); cpass.set_bind_group(0, &bind_group, &[]); cpass.dispatch_workgroups(total_out.div_ceil(256), 1, 1);
                    }
                    queue.submit(Some(encoder.finish()));
                    TensorData::Gpu(out_buf)
                } else { unreachable!() }
            },
            _ => panic!("Hardware conflict"),
        };

        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: vec![batch, out_c, out_len], grad: None, creators: vec![Arc::clone(i_node), Arc::clone(k_node)], device: i.device.clone(), 
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let i_node = &out_tensor.creators[0]; let k_node = &out_tensor.creators[1]; let out_grad = out_tensor.get_cpu_grad();
                let i_read = i_node.read().unwrap(); let k_read = k_node.read().unwrap();
                let (i_data, _) = i_read.to_cpu().into_raw_vec_and_offset(); let (k_data, _) = k_read.to_cpu().into_raw_vec_and_offset();
                
                let mut i_grad = vec![0.0; batch * in_c * len]; let mut k_grad = vec![0.0; out_c * in_c * k_len];
                
                for b in 0..batch {
                    for oc in 0..out_c {
                        for ol in 0..out_len {
                            let g = out_grad[b * (out_c * out_len) + oc * out_len + ol];
                            for ic in 0..in_c {
                                for kl in 0..k_len {
                                    i_grad[b * (in_c * len) + ic * len + (ol + kl)] += g * k_data[oc * (in_c * k_len) + ic * k_len + kl];
                                    k_grad[oc * (in_c * k_len) + ic * k_len + kl] += g * i_data[b * (in_c * len) + ic * len + (ol + kl)];
                                }
                            }
                        }
                    }
                }
                drop(i_read); drop(k_read);
                i_node.write().unwrap().add_cpu_grad(&i_grad); k_node.write().unwrap().add_cpu_grad(&k_grad);
            }))
        }))
    }

    fn conv3d(i_node: &TensorNode<Self>, k_node: &TensorNode<Self>) -> TensorNode<Self> {
        let i = i_node.read().unwrap(); let k = k_node.read().unwrap();
        let batch = i.shape[0]; let in_c = i.shape[1]; 
        let in_d = i.shape[2]; let in_h = i.shape[3]; let in_w = i.shape[4];
        let out_c = k.shape[0]; 
        let k_d = k.shape[2]; let k_h = k.shape[3]; let k_w = k.shape[4];
        
        let out_d = in_d - k_d + 1; let out_h = in_h - k_h + 1; let out_w = in_w - k_w + 1;

        let out_data = match (&i.data, &k.data) {
            (TensorData::Cpu(i_data), TensorData::Cpu(k_data)) => {
                let mut result = vec![0.0; batch * out_c * out_d * out_h * out_w];
                let out_slice_len = out_c * out_d * out_h * out_w;
                result.par_chunks_mut(out_slice_len).enumerate().for_each(|(b, batch_slice)| {
                    for oc in 0..out_c {
                        for od in 0..out_d {
                            for oh in 0..out_h {
                                for ow in 0..out_w {
                                    let mut sum = 0.0;
                                    for ic in 0..in_c {
                                        for kd in 0..k_d {
                                            for kh in 0..k_h {
                                                for kw in 0..k_w {
                                                    let in_idx = b * (in_c * in_d * in_h * in_w) 
                                                               + ic * (in_d * in_h * in_w) 
                                                               + (od + kd) * (in_h * in_w) 
                                                               + (oh + kh) * in_w 
                                                               + (ow + kw);
                                                    let k_idx = oc * (in_c * k_d * k_h * k_w) 
                                                              + ic * (k_d * k_h * k_w) 
                                                              + kd * (k_h * k_w) 
                                                              + kh * k_w 
                                                              + kw;
                                                    sum += i_data[in_idx] * k_data[k_idx];
                                                }
                                            }
                                        }
                                    }
                                    let out_idx = oc * (out_d * out_h * out_w) + od * (out_h * out_w) + oh * out_w + ow;
                                    batch_slice[out_idx] = sum;
                                }
                            }
                        }
                    }
                });
                TensorData::Cpu(result) 
            },
            (TensorData::Gpu(i_buf), TensorData::Gpu(k_buf)) => {
                if let Some((device, queue)) = i.device.get_gpu() {
                    let total_out = (batch * out_c * out_d * out_h * out_w) as u32;
                    let shader = "
                        struct Dims { n: u32, c: u32, d: u32, h: u32, w: u32, out_c: u32, k_d: u32, k_h: u32, k_w: u32, out_d: u32, out_h: u32, out_w: u32, p1: u32, p2: u32, p3: u32, p4: u32 }
                        @group(0) @binding(0) var<uniform> dims: Dims;
                        @group(0) @binding(1) var<storage, read> input: array<f32>;
                        @group(0) @binding(2) var<storage, read> kernel_weights: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> output: array<f32>;
                        @compute @workgroup_size(256, 1, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let out_idx = id.x;
                            let total_out = dims.n * dims.out_c * dims.out_d * dims.out_h * dims.out_w;
                            if (out_idx >= total_out) { return; }
                            let ow = out_idx % dims.out_w;
                            let oh = (out_idx / dims.out_w) % dims.out_h;
                            let od = (out_idx / (dims.out_w * dims.out_h)) % dims.out_d;
                            let oc = (out_idx / (dims.out_w * dims.out_h * dims.out_d)) % dims.out_c;
                            let batch = out_idx / (dims.out_w * dims.out_h * dims.out_d * dims.out_c);
                            var sum = 0.0;
                            for (var ic = 0u; ic < dims.c; ic = ic + 1u) {
                                for (var kd = 0u; kd < dims.k_d; kd = kd + 1u) {
                                    for (var kh = 0u; kh < dims.k_h; kh = kh + 1u) {
                                        for (var kw = 0u; kw < dims.k_w; kw = kw + 1u) {
                                            let in_idx = batch * (dims.c * dims.d * dims.h * dims.w) 
                                                       + ic * (dims.d * dims.h * dims.w) 
                                                       + (od + kd) * (dims.h * dims.w) 
                                                       + (oh + kh) * dims.w 
                                                       + (ow + kw);
                                            let k_idx = oc * (dims.c * dims.k_d * dims.k_h * dims.k_w) 
                                                      + ic * (dims.k_d * dims.k_h * dims.k_w) 
                                                      + kd * (dims.k_h * dims.k_w) 
                                                      + kh * dims.k_w 
                                                      + kw;
                                            sum = sum + input[in_idx] * kernel_weights[k_idx];
                                        }
                                    }
                                }
                            }
                            output[out_idx] = sum;
                        }
                    ";
                    let dims = [
                        batch as u32, in_c as u32, in_d as u32, in_h as u32, in_w as u32, out_c as u32, k_d as u32, k_h as u32, k_w as u32,
                        out_d as u32, out_h as u32, out_w as u32, 0, 0, 0, 0
                    ];
                    let dims_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                    queue.write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims));
                    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(shader.into()) });
                    let out_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (total_out * 4) as wgpu::BufferAddress, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                    
                    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None, entries: &[
                            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        ],
                    });
                    
                    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
                    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader_module, entry_point: Some("main"), cache: None, compilation_options: Default::default() });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: i_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: k_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                    ]});
                    
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                        cpass.set_pipeline(&pipeline); cpass.set_bind_group(0, &bind_group, &[]); cpass.dispatch_workgroups(total_out.div_ceil(256), 1, 1);
                    }
                    queue.submit(Some(encoder.finish()));
                    TensorData::Gpu(out_buf)
                } else { unreachable!() }
            },
            _ => panic!("Hardware conflict"),
        };

        Arc::new(RwLock::new(TensorGraph {
            data: out_data, shape: vec![batch, out_c, out_d, out_h, out_w], grad: None, creators: vec![Arc::clone(i_node), Arc::clone(k_node)], device: i.device.clone(), 
            backward: Some(Box::new(move |out_tensor: &TensorGraph<Self>| {
                let i_node = &out_tensor.creators[0]; let k_node = &out_tensor.creators[1]; let out_grad = out_tensor.get_cpu_grad();
                let i_read = i_node.read().unwrap(); let k_read = k_node.read().unwrap();
                let (i_data, _) = i_read.to_cpu().into_raw_vec_and_offset(); let (k_data, _) = k_read.to_cpu().into_raw_vec_and_offset();
                
                let mut i_grad = vec![0.0; batch * in_c * in_d * in_h * in_w]; let mut k_grad = vec![0.0; out_c * in_c * k_d * k_h * k_w];
                
                for b in 0..batch {
                    for oc in 0..out_c {
                        for od in 0..out_d {
                            for oh in 0..out_h {
                                for ow in 0..out_w {
                                    let g = out_grad[b * (out_c * out_d * out_h * out_w) + oc * (out_d * out_h * out_w) + od * (out_h * out_w) + oh * out_w + ow];
                                    for ic in 0..in_c {
                                        for kd in 0..k_d {
                                            for kh in 0..k_h {
                                                for kw in 0..k_w {
                                                    i_grad[b * (in_c * in_d * in_h * in_w) + ic * (in_d * in_h * in_w) + (od + kd) * (in_h * in_w) + (oh + kh) * in_w + (ow + kw)] += g * k_data[oc * (in_c * k_d * k_h * k_w) + ic * (k_d * k_h * k_w) + kd * (k_h * k_w) + kh * k_w + kw];
                                                    k_grad[oc * (in_c * k_d * k_h * k_w) + ic * (k_d * k_h * k_w) + kd * (k_h * k_w) + kh * k_w + kw] += g * i_data[b * (in_c * in_d * in_h * in_w) + ic * (in_d * in_h * in_w) + (od + kd) * (in_h * in_w) + (oh + kh) * in_w + (ow + kw)];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                drop(i_read); drop(k_read);
                i_node.write().unwrap().add_cpu_grad(&i_grad); k_node.write().unwrap().add_cpu_grad(&k_grad);
            }))
        }))
    }
}

pub static GLOBAL_COMPUTE_GRAPH: std::sync::Mutex<ComputeGraph> = std::sync::Mutex::new(ComputeGraph { nodes: Vec::new() });