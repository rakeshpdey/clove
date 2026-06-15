use std::sync::{Arc, RwLock};
use std::collections::HashSet;
use crate::device::EngineDevice;
use rayon::prelude::*;
use ndarray::Array2;

// Thread-safe smart pointer
pub type Node = Arc<RwLock<Tensor>>;
pub type BackwardOp = Box<dyn Fn(&Tensor) + Send + Sync>;

pub enum TensorData {
    Cpu(Vec<f32>),
    Gpu(wgpu::Buffer),
}

pub struct Tensor {
    pub data: TensorData,
    pub shape: Vec<usize>,
    pub grad: Option<TensorData>, // <-- UPGRADED: Gradients can now live on the GPU!
    pub backward: Option<BackwardOp>,
    pub creators: Vec<Node>,
    pub device: EngineDevice,
}

impl Tensor {
    // =================================================================
    // NEW HARDWARE-SAFE GRADIENT HELPERS
    // =================================================================
    
    /// Safely fetches the gradient slice IF it is on the CPU.
    /// If it is on the GPU, it triggers the "Loud Sentinel" panic.
    pub fn get_cpu_grad(&self) -> &[f32] {
        match &self.grad {
            Some(TensorData::Cpu(g)) => g.as_slice(),
            Some(TensorData::Gpu(_)) => panic!("🔥 AUTOGRAD FATAL: WGPU backward shaders are not fully implemented in v0.1.0! Route engine to CPU."),
            None => panic!("🔥 AUTOGRAD FATAL: Attempted to read gradient before initialization."),
        }
    }

    /// Safely adds a calculated error signal to the tensor's gradient memory.
    pub fn add_cpu_grad(&mut self, new_grad: &[f32]) {
        if self.grad.is_none() {
            // First time receiving an error signal: allocate the memory
            self.grad = Some(TensorData::Cpu(new_grad.to_vec()));
        } else if let Some(TensorData::Cpu(current)) = self.grad.as_mut() {
            // Memory exists: Add the new signal safely using parallel threads
            current.par_iter_mut().zip(new_grad.par_iter()).for_each(|(c, &g)| *c += g);
        } else {
            panic!("🔥 AUTOGRAD FATAL: Hardware Mismatch! Tried to add CPU gradients to a GPU Tensor.");
        }
    }

    // =================================================================
    // INITIALIZATION
    // =================================================================
    pub fn new_cpu(data: Vec<f32>, shape: Vec<usize>) -> Node {
        Arc::new(RwLock::new(Tensor {
            data: TensorData::Cpu(data),
            shape,
            grad: None,
            backward: None,
            creators: vec![],
            device: EngineDevice::Cpu { cores: num_cpus::get() },
        }))
    }

    pub fn new(data_array: Array2<f32>) -> Node {
        let shape = vec![data_array.nrows(), data_array.ncols()];
        let (data, _) = data_array.into_raw_vec_and_offset();
        Self::new_cpu(data, shape)
    }

    pub fn kaiming_random(in_features: usize, out_features: usize) -> Node {
        use rand_distr::{Normal, Distribution};
        let mut rng = rand::rng();
        let std_dev = (2.0 / in_features as f32).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();
        
        let total_elements = in_features * out_features;
        let data: Vec<f32> = (0..total_elements).map(|_| normal.sample(&mut rng)).collect();
        Self::new_cpu(data, vec![in_features, out_features])
    }

    // =================================================================
    // WGPU SHADER DISPATCHERS
    // =================================================================
    pub fn rayon_matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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
    pub fn wgpu_matmul(
        &self, a_buf: &wgpu::Buffer, b_buf: &wgpu::Buffer,
        m: u32, k: u32, n: u32, device: &wgpu::Device, queue: &wgpu::Queue
    ) -> wgpu::Buffer {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("matmul.wgsl").into()),
        });
        
        let c_buffer_size = (m * n * 4) as wgpu::BufferAddress;
        let c_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C (Output)"),
            size: c_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let dims = [m, k, n];
        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dimensions Buffer"),
            size: 12,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
        
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Pipeline"), layout: Some(&compute_pipeline_layout), module: &shader, entry_point: Some("main"), cache: None, compilation_options: Default::default(),
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Bind Group"), layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: c_buf.as_entire_binding() },
            ],
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(n.div_ceil(8), m.div_ceil(8), 1);
        }
        queue.submit(Some(encoder.finish()));
        c_buf
    }

    #[allow(clippy::too_many_arguments)]
    pub fn wgpu_elementwise(
        device: &wgpu::Device, queue: &wgpu::Queue,
        a_buf: &wgpu::Buffer, b_buf: &wgpu::Buffer,
        a_size: u32, b_size: u32, shader_code: &str,
    ) -> wgpu::Buffer {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Elementwise Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

        let c_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (a_size * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dims = [a_size, b_size];
        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dimensions Buffer"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Elementwise Pipeline"), layout: Some(&pipeline_layout), module: &shader, entry_point: Some("main"), cache: None, compilation_options: Default::default(),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: c_buf.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(a_size.div_ceil(256), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        c_buf
    }

    #[allow(clippy::too_many_arguments)]
    pub fn wgpu_add_grad(
        device: &wgpu::Device, queue: &wgpu::Queue,
        grad_out: &wgpu::Buffer,
        grad_a: &wgpu::Buffer,
        grad_b: &wgpu::Buffer,
        a_size: u32, b_size: u32,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Add Grad Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("add_grad.wgsl").into()),
        });

        let dims = [a_size, b_size];
        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Add Grad Dims"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Add Grad Pipeline"), layout: Some(&pipeline_layout), module: &shader, entry_point: Some("main"), cache: None, compilation_options: Default::default(),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grad_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grad_b.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(a_size.div_ceil(256), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    // =================================================================
    // MATH OPERATIONS (FULLY UPGRADED WITH LOCK HELPERS)
    // =================================================================
    pub fn matmul(a_node: &Node, b_node: &Node) -> Node {
        let a = a_node.read().unwrap();
        let b = b_node.read().unwrap();
        
        assert_eq!(a.shape[1], b.shape[0], "Dimension mismatch for matrix multiplication.");
        let m = a.shape[0]; let k = a.shape[1]; let n = b.shape[1];

        let out_data = match (&a.data, &b.data) {
            (TensorData::Cpu(a_vec), TensorData::Cpu(b_vec)) => TensorData::Cpu(a.rayon_matmul(a_vec, b_vec, m, k, n)),
            (TensorData::Gpu(a_buf), TensorData::Gpu(b_buf)) => {
                if let EngineDevice::Gpu { device, queue } = &a.device {
                    TensorData::Gpu(a.wgpu_matmul(a_buf, b_buf, m as u32, k as u32, n as u32, device, queue))
                } else { unreachable!() }
            }
            _ => panic!("Hardware deployment conflict"),
        };

        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: vec![m, n], grad: None,
            creators: vec![Arc::clone(a_node), Arc::clone(b_node)], device: a.device.clone(),
            
            backward: Some(Box::new(|out_tensor: &Tensor| {
                let a_node = &out_tensor.creators[0];
                let b_node = &out_tensor.creators[1];
                let out_grad = out_tensor.get_cpu_grad();

                let a_read = a_node.read().unwrap();
                let b_read = b_node.read().unwrap();
                let m = a_read.shape[0]; let k = a_read.shape[1]; let n = b_read.shape[1];

                let mut a_grad_calc = vec![0.0; m * k];
                let mut b_grad_calc = vec![0.0; k * n];

                if let (TensorData::Cpu(a_data), TensorData::Cpu(b_data)) = (&a_read.data, &b_read.data) {
                    a_grad_calc.par_chunks_mut(k).enumerate().for_each(|(r, row)| {
                        for c in 0..k {
                            let mut sum = 0.0;
                            for i in 0..n { sum += out_grad[r * n + i] * b_data[c * n + i]; }
                            row[c] = sum;
                        }
                    });
                    b_grad_calc.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
                        for c in 0..n {
                            let mut sum = 0.0;
                            for i in 0..m { sum += a_data[i * k + r] * out_grad[i * n + c]; }
                            row[c] = sum;
                        }
                    });
                }
                drop(a_read); drop(b_read);

                a_node.write().unwrap().add_cpu_grad(&a_grad_calc);
                b_node.write().unwrap().add_cpu_grad(&b_grad_calc);
            })),
        }))
    }

    pub fn add(a_node: &Node, b_node: &Node) -> Node {
        let a = a_node.read().unwrap(); let b = b_node.read().unwrap();
        let out_data = match (&a.data, &b.data) {
            (TensorData::Cpu(a_data), TensorData::Cpu(b_data)) => {
                let mut result = vec![0.0; a_data.len()]; let b_len = b_data.len();
                result.par_iter_mut().enumerate().for_each(|(i, res)| { *res = a_data[i] + b_data[i % b_len]; });
                TensorData::Cpu(result)
            },
            (TensorData::Gpu(a_buf), TensorData::Gpu(b_buf)) => {
                if let EngineDevice::Gpu { device, queue } = &a.device {
                    let a_size = a.shape.iter().product::<usize>() as u32;
                    let b_size = b.shape.iter().product::<usize>() as u32;
                    TensorData::Gpu(Self::wgpu_elementwise(
                        device, queue, a_buf, b_buf, a_size, b_size, include_str!("add.wgsl")
                    ))
                } else { unreachable!() }
            },
            _ => panic!("Hardware deployment conflict: Cannot mix CPU and GPU tensors"),
        };
        
        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: a.shape.clone(), grad: None,
            creators: vec![Arc::clone(a_node), Arc::clone(b_node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &Tensor| {
                let a_node = &out_tensor.creators[0];
                let b_node = &out_tensor.creators[1];
                match out_tensor.grad.as_ref().expect("Missing gradient") {
                    TensorData::Cpu(out_grad) => {
                        a_node.write().unwrap().add_cpu_grad(out_grad);

                        let b_len = b_node.read().unwrap().shape.iter().product();
                        let mut b_grad_calc = vec![0.0; b_len];
                        for (i, &g) in out_grad.iter().enumerate() { b_grad_calc[i % b_len] += g; }
                        b_node.write().unwrap().add_cpu_grad(&b_grad_calc);
                    }
                    TensorData::Gpu(out_grad_buf) => {
                        if let EngineDevice::Gpu { device, queue } = &out_tensor.device {
                            let a_size = a_node.read().unwrap().shape.iter().product::<usize>() as u32;
                            let b_size = b_node.read().unwrap().shape.iter().product::<usize>() as u32;

                            let init_gpu_grad = |node: &Node, size: u32| {
                                let mut n = node.write().unwrap();
                                if n.grad.is_none() {
                                    let buf = device.create_buffer(&wgpu::BufferDescriptor {
                                        label: Some("GPU Grad Buffer"),
                                        size: (size * 4) as u64,
                                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                                        mapped_at_creation: false,
                                    });
                                    let zeros = vec![0.0f32; size as usize];
                                    queue.write_buffer(&buf, 0, bytemuck::cast_slice(&zeros));
                                    n.grad = Some(TensorData::Gpu(buf));
                                }
                            };
                            init_gpu_grad(a_node, a_size);
                            init_gpu_grad(b_node, b_size);

                            let a_read = a_node.read().unwrap();
                            let b_read = b_node.read().unwrap();
                            let a_grad_buf = if let Some(TensorData::Gpu(b)) = &a_read.grad { b } else { unreachable!() };
                            let b_grad_buf = if let Some(TensorData::Gpu(b)) = &b_read.grad { b } else { unreachable!() };
                            Self::wgpu_add_grad(device, queue, out_grad_buf, a_grad_buf, b_grad_buf, a_size, b_size);
                        } else {
                            unreachable!()
                        }
                    }
                }
            }))
        }))
    }

    pub fn mul(a_node: &Node, b_node: &Node) -> Node {
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
                if let EngineDevice::Gpu { device, queue } = &a.device {
                    let a_size = a.shape.iter().product::<usize>() as u32;
                    let b_size = b.shape.iter().product::<usize>() as u32;
                    TensorData::Gpu(Self::wgpu_elementwise(
                        device, queue, a_buf, b_buf, a_size, b_size, include_str!("mul.wgsl")
                    ))
                } else { unreachable!() }
            },
            _ => panic!("Hardware deployment conflict: Cannot mix CPU and GPU tensors"),
        };
        
        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: a.shape.clone(), grad: None,
            creators: vec![Arc::clone(a_node), Arc::clone(b_node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &Tensor| {
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
                        if b_is_scalar { b_grad_calc[0] += out_grad[i] * a_val; } 
                        else { b_grad_calc[i] += out_grad[i] * a_val; }
                    }
                }
                drop(a_read); drop(b_read);

                a_node.write().unwrap().add_cpu_grad(&a_grad_calc);
                b_node.write().unwrap().add_cpu_grad(&b_grad_calc);
            })), 
        }))
    }

    pub fn embedding(weights_node: &Node, indices_matrix: &Array2<f32>) -> Node {
        let weights = weights_node.read().unwrap();
        let vocab_size = weights.shape[0]; let hidden_size = weights.shape[1];
        let seq_len = indices_matrix.nrows();
        
        let indices: Vec<usize> = indices_matrix.iter().map(|&x| x as usize).collect();
        let out_data = match &weights.data {
            TensorData::Cpu(w_data) => {
                let mut result = vec![0.0; seq_len * hidden_size];
                result.par_chunks_mut(hidden_size).enumerate().for_each(|(i, out_row)| {
                    let token_id = indices[i];
                    if token_id < vocab_size {
                        let row_start = token_id * hidden_size;
                        out_row.copy_from_slice(&w_data[row_start..(row_start + hidden_size)]);
                    }
                });
                TensorData::Cpu(result)
            },
            _ => unimplemented!("GPU Embedding pipeline not yet linked"),
        };

        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: vec![seq_len, hidden_size], grad: None,
            creators: vec![Arc::clone(weights_node)], device: weights.device.clone(),
            backward: Some(Box::new(move |out_tensor: &Tensor| {
                let weights_node = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad();
                
                let hidden_size = weights_node.read().unwrap().shape[1];
                let vocab_size = weights_node.read().unwrap().shape[0];
                let mut w_grad_calc = vec![0.0; vocab_size * hidden_size];
                
                for (seq_idx, &token_id) in indices.iter().enumerate() {
                    let row_start = token_id * hidden_size;
                    let grad_start = seq_idx * hidden_size;
                    for i in 0..hidden_size { w_grad_calc[row_start + i] += out_grad[grad_start + i]; }
                }
                
                weights_node.write().unwrap().add_cpu_grad(&w_grad_calc);
            }))
        }))
    }

    pub fn transpose(node: &Node) -> Node {
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
            _ => unimplemented!("GPU Transpose pipeline not yet linked"),
        };

        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: vec![cols, rows], grad: None,
            creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(move |out_tensor: &Tensor| {
                let parent = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad();
                let mut p_grad = vec![0.0; rows * cols];
                
                for r in 0..rows {
                    for c in 0..cols { p_grad[c * rows + r] = out_grad[r * cols + c]; }
                }
                
                parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }

    pub fn mul_scalar(node: &Node, scalar: f32) -> Node {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &val)| { *res = val * scalar; });
                TensorData::Cpu(result)
            },
            _ => unimplemented!("GPU mul_scalar pipeline not yet linked"),
        };

        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: a.shape.clone(), grad: None,
            creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(move |out_tensor: &Tensor| {
                let parent = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad();
                let mut p_grad = vec![0.0; out_grad.len()];
                for i in 0..out_grad.len() { p_grad[i] = out_grad[i] * scalar; }
                
                parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }

    pub fn cross_entropy(logits_node: &Node, targets: &Array2<f32>) -> Node {
        let logits = logits_node.read().unwrap();
        let rows = logits.shape[0]; let cols = logits.shape[1];
        let targets_vec: Vec<f32> = targets.iter().cloned().collect();
        
        let out_data = match &logits.data {
            TensorData::Cpu(l_data) => {
                let mut total_loss = 0.0;
                for i in 0..rows {
                    let row_start = i * cols;
                    let row_logits = &l_data[row_start..(row_start + cols)];
                    let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0;
                    for &val in row_logits { sum_exp += (val - max_logit).exp(); }
                    
                    let mut target_idx = 0;
                    for j in 0..cols {
                        if targets[[i, j]] > 0.5 { target_idx = j; break; }
                    }
                    total_loss -= (row_logits[target_idx] - max_logit) - sum_exp.ln();
                }
                TensorData::Cpu(vec![total_loss / (rows as f32)])
            },
            _ => unimplemented!("GPU Cross Entropy pipeline not yet linked"),
        };

        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: vec![1, 1], grad: None,
            creators: vec![Arc::clone(logits_node)], device: logits.device.clone(),
            backward: Some(Box::new(move |out_tensor: &Tensor| {
                let logits_node = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad()[0];

                let logits_read = logits_node.read().unwrap();
                let rows = logits_read.shape[0]; let cols = logits_read.shape[1];
                let mut grad_calc = vec![0.0; rows * cols];

                if let TensorData::Cpu(l_data) = &logits_read.data {
                    for i in 0..rows {
                        let row_start = i * cols;
                        let row_logits = &l_data[row_start..(row_start + cols)];
                        let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        
                        let mut sum_exp = 0.0;
                        let mut exps = vec![0.0; cols];
                        for j in 0..cols {
                            let e = (row_logits[j] - max_logit).exp();
                            exps[j] = e;
                            sum_exp += e;
                        }
                        
                        for j in 0..cols {
                            let prob = exps[j] / sum_exp;
                            let target = targets_vec[row_start + j];
                            grad_calc[row_start + j] = (prob - target) * out_grad / (rows as f32);
                        }
                    }
                }
                drop(logits_read);
                
                logits_node.write().unwrap().add_cpu_grad(&grad_calc);
            }))
        }))
    }

    pub fn gelu(node: &Node) -> Node {
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
            _ => unimplemented!("GPU GELU pipeline not yet linked"),
        };

        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: a.shape.clone(), grad: None,
            creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &Tensor| {
                let parent = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad();
                let p_read = parent.read().unwrap();
                let p_data = if let TensorData::Cpu(d) = &p_read.data { d } else { return };
                
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

    pub fn dropout(node: &Node, rate: f32) -> Node {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                let scale = 1.0 / (1.0 - rate);
                result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &x)| {
                    let drop: f32 = rand::random();
                    if drop >= rate { *res = x * scale; } else { *res = 0.0; }
                });
                TensorData::Cpu(result)
            },
            _ => unimplemented!("GPU Dropout pipeline not yet linked"),
        };

        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: a.shape.clone(), grad: None,
            creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(move |out_tensor: &Tensor| {
                let parent = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad();
                let out_data = if let TensorData::Cpu(d) = &out_tensor.data { d } else { return };
                
                let mut p_grad = vec![0.0; out_grad.len()];
                let scale = 1.0 / (1.0 - rate);
                for i in 0..out_grad.len() {
                    if out_data[i] == 0.0 { p_grad[i] = 0.0; } else { p_grad[i] = out_grad[i] * scale; }
                }
                
                parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }

    pub fn softmax(node: &Node) -> Node {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                let cols = a.shape[1];
                result.par_chunks_mut(cols).enumerate().for_each(|(i, row_out)| {
                    let row_in = &a_data[i * cols .. (i + 1) * cols];
                    let max_val = row_in.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0;
                    for j in 0..cols {
                        let exp_val = (row_in[j] - max_val).exp();
                        row_out[j] = exp_val;
                        sum += exp_val;
                    }
                    for val in row_out.iter_mut() { *val /= sum; }
                });
                TensorData::Cpu(result)
            },
            _ => unimplemented!("GPU Softmax pipeline not yet linked"),
        };

        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: a.shape.clone(), grad: None,
            creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &Tensor| {
                let parent = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad();
                let out_data = if let TensorData::Cpu(d) = &out_tensor.data { d } else { return };
                
                let mut p_grad = vec![0.0; out_grad.len()];
                for i in 0..out_grad.len() { p_grad[i] = out_grad[i] * out_data[i] * (1.0 - out_data[i]); }
                
                parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }
    
    pub fn relu(node: &Node) -> Node {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &x)| {
                    *res = if x > 0.0 { x } else { 0.0 };
                });
                TensorData::Cpu(result)
            },
            _ => unimplemented!("GPU ReLU not linked"),
        };
        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: a.shape.clone(), grad: None,
            creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &Tensor| {
                let parent = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad();
                let out_data = if let TensorData::Cpu(d) = &out_tensor.data { d } else { return };
                let mut p_grad = vec![0.0; out_grad.len()];
                for i in 0..out_grad.len() { p_grad[i] = if out_data[i] > 0.0 { out_grad[i] } else { 0.0 }; }
                
                parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }

    pub fn mse(pred_node: &Node, target: &Array2<f32>) -> Node {
        let pred = pred_node.read().unwrap();
        let target_vec: Vec<f32> = target.iter().cloned().collect();
        let out_data = match &pred.data {
            TensorData::Cpu(p_data) => {
                let mut sum = 0.0;
                for i in 0..p_data.len() { sum += (p_data[i] - target_vec[i]).powi(2); }
                TensorData::Cpu(vec![sum / p_data.len() as f32])
            },
            _ => unimplemented!("GPU MSE not linked"),
        };
        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: vec![1, 1], grad: None,
            creators: vec![Arc::clone(pred_node)], device: pred.device.clone(),
            backward: Some(Box::new(move |out_tensor: &Tensor| {
                let pred_node = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad()[0];
                let pred_read = pred_node.read().unwrap();
                let mut grad_calc = vec![0.0; target_vec.len()];
                if let TensorData::Cpu(p_data) = &pred_read.data {
                    let n = p_data.len() as f32;
                    for i in 0..p_data.len() { grad_calc[i] = 2.0 * (p_data[i] - target_vec[i]) * out_grad / n; }
                }
                drop(pred_read);
                
                pred_node.write().unwrap().add_cpu_grad(&grad_calc);
            }))
        }))
    }

    pub fn sub(a_node: &Node, b_node: &Node) -> Node {
        let a = a_node.read().unwrap(); let b = b_node.read().unwrap();
        let out_data = match (&a.data, &b.data) {
            (TensorData::Cpu(a_data), TensorData::Cpu(b_data)) => {
                let mut result = vec![0.0; a_data.len()]; let b_len = b_data.len();
                result.par_iter_mut().enumerate().for_each(|(i, res)| { *res = a_data[i] - b_data[i % b_len]; });
                TensorData::Cpu(result)
            },
            _ => unimplemented!("GPU Sub not linked"),
        };
        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: a.shape.clone(), grad: None,
            creators: vec![Arc::clone(a_node), Arc::clone(b_node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &Tensor| {
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

    pub fn sin(node: &Node) -> Node {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &x)| { *res = x.sin(); });
                TensorData::Cpu(result)
            },
            _ => unimplemented!(),
        };
        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: a.shape.clone(), grad: None,
            creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &Tensor| {
                let parent = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad();
                let p_read = parent.read().unwrap();
                let p_data = if let TensorData::Cpu(d) = &p_read.data { d } else { return };
                let mut p_grad = vec![0.0; out_grad.len()];
                for i in 0..out_grad.len() { p_grad[i] = out_grad[i] * p_data[i].cos(); }
                drop(p_read);
                
                parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }

    pub fn cos(node: &Node) -> Node {
        let a = node.read().unwrap();
        let out_data = match &a.data {
            TensorData::Cpu(a_data) => {
                let mut result = vec![0.0; a_data.len()];
                result.par_iter_mut().zip(a_data.par_iter()).for_each(|(res, &x)| { *res = x.cos(); });
                TensorData::Cpu(result)
            },
            _ => unimplemented!(),
        };
        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: a.shape.clone(), grad: None,
            creators: vec![Arc::clone(node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &Tensor| {
                let parent = &out_tensor.creators[0];
                let out_grad = out_tensor.get_cpu_grad();
                let p_read = parent.read().unwrap();
                let p_data = if let TensorData::Cpu(d) = &p_read.data { d } else { return };
                let mut p_grad = vec![0.0; out_grad.len()];
                for i in 0..out_grad.len() { p_grad[i] = out_grad[i] * -p_data[i].sin(); }
                drop(p_read);
                
                parent.write().unwrap().add_cpu_grad(&p_grad);
            }))
        }))
    }

    pub fn conv2d(image_node: &Node, kernel_node: &Node) -> Node {
        let image = image_node.read().unwrap(); let kernel = kernel_node.read().unwrap();
        let i_rows = image.shape[0]; let i_cols = image.shape[1];
        let k_rows = kernel.shape[0]; let k_cols = kernel.shape[1];
        let out_rows = i_rows - k_rows + 1; let out_cols = i_cols - k_cols + 1;

        let out_data = match (&image.data, &kernel.data) {
            (TensorData::Cpu(i_data), TensorData::Cpu(k_data)) => {
                let mut result = vec![0.0; out_rows * out_cols];
                for r in 0..out_rows {
                    for c in 0..out_cols {
                        let mut sum = 0.0;
                        for kr in 0..k_rows {
                            for kc in 0..k_cols { sum += i_data[(r + kr) * i_cols + (c + kc)] * k_data[kr * k_cols + kc]; }
                        }
                        result[r * out_cols + c] = sum;
                    }
                }
                TensorData::Cpu(result)
            },
            _ => unimplemented!("GPU Conv2d not linked"),
        };
        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: vec![out_rows, out_cols], grad: None,
            creators: vec![Arc::clone(image_node), Arc::clone(kernel_node)], device: image.device.clone(),
            backward: Some(Box::new(move |out_tensor: &Tensor| {
                let image_node = &out_tensor.creators[0]; let kernel_node = &out_tensor.creators[1];
                let out_grad = out_tensor.get_cpu_grad();
                
                let image = image_node.read().unwrap(); let kernel = kernel_node.read().unwrap();
                let i_data = if let TensorData::Cpu(d) = &image.data { d } else { return };
                let k_data = if let TensorData::Cpu(d) = &kernel.data { d } else { return };

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
                
                image_node.write().unwrap().add_cpu_grad(&i_grad);
                kernel_node.write().unwrap().add_cpu_grad(&k_grad);
            }))
        }))
    }

    pub fn layer_norm(node: &Node, gamma_node: &Node, beta_node: &Node) -> Node {
        let a = node.read().unwrap(); let gamma = gamma_node.read().unwrap(); let beta = beta_node.read().unwrap();
        let out_data = match (&a.data, &gamma.data, &beta.data) {
            (TensorData::Cpu(a_data), TensorData::Cpu(g_data), TensorData::Cpu(b_data)) => {
                let mut result = vec![0.0; a_data.len()]; let cols = a.shape[1];
                result.par_chunks_mut(cols).enumerate().for_each(|(i, row_out)| {
                    let row_in = &a_data[i * cols .. (i + 1) * cols];
                    let mean = row_in.iter().sum::<f32>() / cols as f32;
                    let var = row_in.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / cols as f32;
                    let std_dev = (var + 1e-5).sqrt();
                    for j in 0..cols { row_out[j] = ((row_in[j] - mean) / std_dev) * g_data[j] + b_data[j]; }
                });
                TensorData::Cpu(result)
            },
            _ => unimplemented!("GPU LayerNorm pipeline not yet linked"),
        };

        Arc::new(RwLock::new(Tensor {
            data: out_data, shape: a.shape.clone(), grad: None,
            creators: vec![Arc::clone(node), Arc::clone(gamma_node), Arc::clone(beta_node)], device: a.device.clone(),
            backward: Some(Box::new(|out_tensor: &Tensor| {
                let parent = &out_tensor.creators[0]; let gamma_node = &out_tensor.creators[1]; let beta_node = &out_tensor.creators[2];
                let out_grad = out_tensor.get_cpu_grad();

                let gamma_read = gamma_node.read().unwrap();
                let g_data = if let TensorData::Cpu(g) = &gamma_read.data { g } else { return };
                
                let mut p_grad = vec![0.0; out_grad.len()]; let cols = gamma_read.shape[0];
                for i in 0..out_grad.len() { p_grad[i] = out_grad[i] * g_data[i % cols]; }
                
                parent.write().unwrap().add_cpu_grad(&p_grad);

                let mut b_grad = vec![0.0; cols];
                for i in 0..out_grad.len() { b_grad[i % cols] += out_grad[i]; }
                
                beta_node.write().unwrap().add_cpu_grad(&b_grad);
            }))
        }))
    }

    // =================================================================
    // BACKWARD & TAPE RECORDER
    // =================================================================
    fn build_topo(v: &Node, topo: &mut Vec<Node>, visited: &mut HashSet<usize>) {
        let ptr = Arc::as_ptr(v) as usize;
        if !visited.contains(&ptr) {
            visited.insert(ptr);
            for child in &v.read().unwrap().creators { Self::build_topo(child, topo, visited); }
            topo.push(Arc::clone(v));
        }
    }

    pub fn backward(node: &Node) {
        let mut topo = Vec::new(); let mut visited = HashSet::new();
        Self::build_topo(node, &mut topo, &mut visited);

        {
            let mut root = node.write().unwrap();
            let total_elements = root.shape.iter().product();
            root.grad = Some(TensorData::Cpu(vec![1.0; total_elements])); // <-- UPGRADED INITIALIZER
        } 

        for v in topo.into_iter().rev() {
            // Read closure mapping, drop lock instantly, then execute it
            let backward_closure = {
                let v_read = v.read().unwrap();
                v_read.backward.as_ref().map(|b| {
                    let ptr: *const (dyn Fn(&Tensor) + Send + Sync) = b.as_ref();
                    ptr
                })
            };
            if let Some(bwd_ptr) = backward_closure {
                let v_read = v.read().unwrap();
                unsafe { (*bwd_ptr)(&v_read); }
            }
        }
    }

    /// Extracts the data from VRAM back to standard RAM.
    /// If the data is already on the CPU, it just returns a copy.
    pub fn to_cpu(&self) -> Array2<f32> {
        let rows = self.shape[0];
        let cols = self.shape[1];

        match &self.data {
            TensorData::Cpu(vec) => {
                Array2::from_shape_vec((rows, cols), vec.clone())
                    .expect("CPU Shape mismatch during extraction")
            }
            TensorData::Gpu(buffer) => {
                if let EngineDevice::Gpu { device, queue } = &self.device {
                    let size = buffer.size();
                    
                    // 1. Create a "Staging Buffer" with MAP_READ permissions
                    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("VRAM to RAM Staging Buffer"),
                        size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    // 2. Command the GPU to copy the data from the Storage Buffer to the Staging Buffer
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
                    queue.submit(Some(encoder.finish()));

                    // 3. Request memory mapping (Async)
                    let buffer_slice = staging_buffer.slice(..);
                    let (sender, receiver) = std::sync::mpsc::channel();
                    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

                    // 4. Force the CPU to wait for the GPU to finish its operations
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    receiver.recv().unwrap().expect("Failed to read WGPU Buffer!");

                    // 5. Extract the raw bytes and cast them back to f32 floats
                    let mapped_data = buffer_slice.get_mapped_range();
                    let extracted_floats: Vec<f32> = bytemuck::cast_slice(&mapped_data).to_vec();

                    // 6. Clean up GPU memory locks
                    drop(mapped_data);
                    staging_buffer.unmap();

                    Array2::from_shape_vec((rows, cols), extracted_floats)
                        .expect("GPU Shape mismatch during extraction")
                } else {
                    unreachable!()
                }
            }
        }
    }

    pub fn clip_gradients(&mut self) {
        if let Some(TensorData::Cpu(ref mut gradients)) = self.grad { 
            gradients.par_iter_mut().for_each(|g| { *g = g.clamp(-1.0, 1.0); }); 
        }
    }
}