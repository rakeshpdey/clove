use clove::backend::{Backend, WgpuBackend};
use clove::lazy::{LazyBackend, compile};
use ndarray::Array2;
use std::collections::HashMap;

// --- WGPU HARDWARE HELPERS ---
async fn init_wgpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let is_ci = std::env::var("CI").is_ok();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: is_ci,
        })
        .await
        .expect("Failed to find WebGPU adapter!");
    adapter
        .request_device(&Default::default())
        .await
        .expect("Failed to create WebGPU device!")
}

fn array_to_buffer(device: &wgpu::Device, queue: &wgpu::Queue, arr: &Array2<f32>) -> wgpu::Buffer {
    let floats = arr.as_slice().unwrap();
    let size = (floats.len() * 4) as wgpu::BufferAddress;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Input Buffer"),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(floats));
    buffer
}

async fn buffer_to_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) -> Array2<f32> {
    let size = buffer.size();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Download Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let mut floats: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();

    floats.truncate(rows * cols);
    Array2::from_shape_vec((rows, cols), floats).unwrap()
}

// --- MATH PARITY ASSERTION ---
fn assert_tensors_match(op_name: &str, eager: &Array2<f32>, lazy: &Array2<f32>, tol: f32) {
    assert_eq!(
        eager.shape(),
        lazy.shape(),
        "[{}] Shape Mismatch! Eager: {:?}, Lazy: {:?}",
        op_name,
        eager.shape(),
        lazy.shape()
    );
    for (e, l) in eager.iter().zip(lazy.iter()) {
        assert!(
            (e - l).abs() < tol,
            "[{}] Value Mismatch! Eager: {:.6}, Lazy: {:.6}",
            op_name,
            e,
            l
        );
    }
    println!("[{}] Passed Parity Check!", op_name);
}

// to prevent Rust crate versioning errors from interrupting the tests.
fn generate_random_array(rows: usize, cols: usize) -> Array2<f32> {
    let mut data = Vec::with_capacity(rows * cols);
    let mut seed: u32 = 42;
    for _ in 0..(rows * cols) {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let val = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
        data.push(val);
    }
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

fn generate_positive_array(rows: usize, cols: usize) -> Array2<f32> {
    let mut data = Vec::with_capacity(rows * cols);
    let mut seed: u32 = 12345;
    for _ in 0..(rows * cols) {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let val = (seed as f32 / u32::MAX as f32) * 0.9 + 0.1;
        data.push(val);
    }
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

// BINARY OPERATIONS

#[test]
fn test_addition_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);
        let arr_b = generate_random_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_b = WgpuBackend::new(arr_b.clone());
        let eager_out = WgpuBackend::add(&eager_a, &eager_b);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let lazy_b = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);

        let compiled = compile(
            &device,
            |i| LazyBackend::add(i[0], i[1]),
            &[&lazy_a, &lazy_b],
        );
        let buf_a = array_to_buffer(&device, &queue, &arr_a);
        let buf_b = array_to_buffer(&device, &queue, &arr_b);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a, &buf_b], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 16, 16).await;

        assert_tensors_match("Add", &eager_result, &lazy_result, 1e-5);
    });
}

#[test]
fn test_subtraction_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);
        let arr_b = generate_random_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_b = WgpuBackend::new(arr_b.clone());
        let eager_out = WgpuBackend::sub(&eager_a, &eager_b);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let lazy_b = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);

        let compiled = compile(
            &device,
            |i| LazyBackend::sub(i[0], i[1]),
            &[&lazy_a, &lazy_b],
        );
        let buf_a = array_to_buffer(&device, &queue, &arr_a);
        let buf_b = array_to_buffer(&device, &queue, &arr_b);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a, &buf_b], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 16, 16).await;

        assert_tensors_match("Sub", &eager_result, &lazy_result, 1e-5);
    });
}

#[test]
fn test_multiplication_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);
        let arr_b = generate_random_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_b = WgpuBackend::new(arr_b.clone());
        let eager_out = WgpuBackend::mul(&eager_a, &eager_b);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let lazy_b = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);

        let compiled = compile(
            &device,
            |i| LazyBackend::mul(i[0], i[1]),
            &[&lazy_a, &lazy_b],
        );
        let buf_a = array_to_buffer(&device, &queue, &arr_a);
        let buf_b = array_to_buffer(&device, &queue, &arr_b);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a, &buf_b], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 16, 16).await;

        assert_tensors_match("Mul", &eager_result, &lazy_result, 1e-5);
    });
}

#[test]
fn test_matmul_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 32);
        let arr_b = generate_random_array(32, 8);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_b = WgpuBackend::new(arr_b.clone());
        let eager_out = WgpuBackend::matmul(&eager_a, &eager_b);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 16 * 32], vec![16, 32]);
        let lazy_b = LazyBackend::new_cpu(vec![0.0; 32 * 8], vec![32, 8]);

        let compiled = compile(
            &device,
            |i| LazyBackend::matmul(i[0], i[1]),
            &[&lazy_a, &lazy_b],
        );
        let buf_a = array_to_buffer(&device, &queue, &arr_a);
        let buf_b = array_to_buffer(&device, &queue, &arr_b);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a, &buf_b], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 16, 8).await;

        assert_tensors_match("MatMul", &eager_result, &lazy_result, 1e-4);
    });
}

// UNARY OPERATIONS

#[test]
fn test_relu_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::relu(&eager_a);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let compiled = compile(&device, |i| LazyBackend::relu(i[0]), &[&lazy_a]);
        let buf_a = array_to_buffer(&device, &queue, &arr_a);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 16, 16).await;

        assert_tensors_match("ReLU", &eager_result, &lazy_result, 1e-5);
    });
}

#[test]
fn test_gelu_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::gelu(&eager_a);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let compiled = compile(&device, |i| LazyBackend::gelu(i[0]), &[&lazy_a]);
        let buf_a = array_to_buffer(&device, &queue, &arr_a);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 16, 16).await;

        assert_tensors_match("GELU", &eager_result, &lazy_result, 1e-4);
    });
}

#[test]
fn test_sin_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::sin(&eager_a);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let compiled = compile(&device, |i| LazyBackend::sin(i[0]), &[&lazy_a]);
        let buf_a = array_to_buffer(&device, &queue, &arr_a);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 16, 16).await;

        assert_tensors_match("Sin", &eager_result, &lazy_result, 1e-5);
    });
}

#[test]
fn test_cos_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::cos(&eager_a);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let compiled = compile(&device, |i| LazyBackend::cos(i[0]), &[&lazy_a]);
        let buf_a = array_to_buffer(&device, &queue, &arr_a);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 16, 16).await;

        assert_tensors_match("Cos", &eager_result, &lazy_result, 1e-5);
    });
}

#[test]
fn test_sigmoid_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::sigmoid(&eager_a);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let compiled = compile(&device, |i| LazyBackend::sigmoid(i[0]), &[&lazy_a]);
        let buf_a = array_to_buffer(&device, &queue, &arr_a);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 16, 16).await;

        assert_tensors_match("Sigmoid", &eager_result, &lazy_result, 1e-5);
    });
}

#[test]
fn test_tanh_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::tanh(&eager_a);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let compiled = compile(&device, |i| LazyBackend::tanh(i[0]), &[&lazy_a]);
        let buf_a = array_to_buffer(&device, &queue, &arr_a);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 16, 16).await;

        assert_tensors_match("Tanh", &eager_result, &lazy_result, 1e-5);
    });
}

#[test]
fn test_softmax_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(8, 64);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::softmax(&eager_a);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 8 * 64], vec![8, 64]);
        let compiled = compile(&device, |i| LazyBackend::softmax(i[0]), &[&lazy_a]);
        let buf_a = array_to_buffer(&device, &queue, &arr_a);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 8, 64).await;

        assert_tensors_match("Softmax", &eager_result, &lazy_result, 1e-5);
    });
}

#[test]
fn test_scalar_mul_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);
        let scalar = std::f32::consts::PI;

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::mul_scalar(&eager_a, scalar);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let compiled = compile(
            &device,
            |i| LazyBackend::mul_scalar(i[0], scalar),
            &[&lazy_a],
        );
        let buf_a = array_to_buffer(&device, &queue, &arr_a);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 16, 16).await;

        assert_tensors_match("ScalarMul", &eager_result, &lazy_result, 1e-5);
    });
}

// SHAPE & ARCHITECTURE OPERATIONS

#[test]
fn test_transpose_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 32);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::transpose(&eager_a);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 16 * 32], vec![16, 32]);
        let compiled = compile(&device, |i| LazyBackend::transpose(i[0]), &[&lazy_a]);
        let buf_a = array_to_buffer(&device, &queue, &arr_a);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 32, 16).await;

        assert_tensors_match("Transpose", &eager_result, &lazy_result, 1e-5);
    });
}

#[test]
fn test_flatten_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::flatten(&eager_a);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let compiled = compile(&device, |i| LazyBackend::flatten(i[0]), &[&lazy_a]);
        let buf_a = array_to_buffer(&device, &queue, &arr_a);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 1, 256).await;

        assert_tensors_match("Flatten", &eager_result, &lazy_result, 1e-5);
    });
}

#[test]
fn test_layer_norm_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(8, 64);
        let arr_g = generate_random_array(1, 64);
        let arr_b = generate_random_array(1, 64);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_g = WgpuBackend::new(arr_g.clone());
        let eager_b = WgpuBackend::new(arr_b.clone());
        let eager_out = WgpuBackend::layer_norm(&eager_a, &eager_g, &eager_b);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 8 * 64], vec![8, 64]);
        let lazy_g = LazyBackend::new_cpu(vec![0.0; 64], vec![1, 64]);
        let lazy_b = LazyBackend::new_cpu(vec![0.0; 64], vec![1, 64]);

        let compiled = compile(
            &device,
            |i| LazyBackend::layer_norm(i[0], i[1], i[2]),
            &[&lazy_a, &lazy_g, &lazy_b],
        );
        let buf_a = array_to_buffer(&device, &queue, &arr_a);
        let buf_g = array_to_buffer(&device, &queue, &arr_g);
        let buf_b = array_to_buffer(&device, &queue, &arr_b);

        let out_bufs =
            compiled.execute(&device, &queue, &[&buf_a, &buf_g, &buf_b], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 8, 64).await;

        assert_tensors_match("LayerNorm", &eager_result, &lazy_result, 1e-4);
    });
}

#[test]
fn test_dropout_eval_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::dropout(&eager_a, 0.0);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let compiled = compile(&device, |i| LazyBackend::dropout(i[0], 0.0), &[&lazy_a]);
        let buf_a = array_to_buffer(&device, &queue, &arr_a);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 16, 16).await;

        assert_tensors_match("Dropout (Rate=0.0)", &eager_result, &lazy_result, 1e-5);
    });
}

#[test]
fn test_rope_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(4, 64);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::rope(&eager_a, 0, 16);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 4 * 64], vec![4, 64]);
        let compiled = compile(&device, |i| LazyBackend::rope(i[0], 0, 16), &[&lazy_a]);
        let buf_a = array_to_buffer(&device, &queue, &arr_a);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 4, 64).await;

        assert_tensors_match("RoPE", &eager_result, &lazy_result, 1e-4);
    });
}

// LOSS FUNCTIONS

#[test]
fn test_mse_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);
        let arr_b = generate_random_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::mse(&eager_a, &arr_b);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let compiled = compile(&device, |i| LazyBackend::mse(i[0], &arr_b), &[&lazy_a]);
        let buf_a = array_to_buffer(&device, &queue, &arr_a);
        let buf_b = array_to_buffer(&device, &queue, &arr_b);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a, &buf_b], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 1, 1).await;

        assert_tensors_match("MSE", &eager_result, &lazy_result, 1e-4);
    });
}

#[test]
fn test_huber_loss_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);
        let arr_b = generate_random_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::huber_loss(&eager_a, &arr_b, 1.0);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let compiled = compile(
            &device,
            |i| LazyBackend::huber_loss(i[0], &arr_b, 1.0),
            &[&lazy_a],
        );
        let buf_a = array_to_buffer(&device, &queue, &arr_a);
        let buf_b = array_to_buffer(&device, &queue, &arr_b);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a, &buf_b], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 1, 1).await;

        assert_tensors_match("Huber Loss", &eager_result, &lazy_result, 1e-4);
    });
}

#[test]
fn test_bce_parity() {
    pollster::block_on(async {
        let (device, queue) = init_wgpu().await;
        let arr_a = generate_random_array(16, 16);
        let arr_b = generate_positive_array(16, 16);

        let eager_a = WgpuBackend::new(arr_a.clone());
        let eager_out = WgpuBackend::bce_with_logits(&eager_a, &arr_b);
        let eager_result = WgpuBackend::to_cpu(&eager_out.read().unwrap());

        let lazy_a = LazyBackend::new_cpu(vec![0.0; 256], vec![16, 16]);
        let compiled = compile(
            &device,
            |i| LazyBackend::bce_with_logits(i[0], &arr_b),
            &[&lazy_a],
        );
        let buf_a = array_to_buffer(&device, &queue, &arr_a);
        let buf_b = array_to_buffer(&device, &queue, &arr_b);

        let out_bufs = compiled.execute(&device, &queue, &[&buf_a, &buf_b], &HashMap::new());
        let lazy_result = buffer_to_array(&device, &queue, &out_bufs[0], 1, 1).await;

        assert_tensors_match("BCE With Logits", &eager_result, &lazy_result, 1e-4);
    });
}
