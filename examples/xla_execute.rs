use organon::lazy::{LazyBackend, compile};
use organon::backend::Backend;

async fn run_jit_execution() {
    println!("====================================================");
    println!("     ORGANON XLA: MATMUL FUSION & CACHE TEST        ");
    println!("====================================================\n");

    let instance = wgpu::Instance::default();
    
    // Fetch all available adapters and dynamically hunt for a capable one
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let adapter = adapters.into_iter().find(|a| {
        let limits = a.limits();
        // We need at least 4 storage buffers and compute capabilities
        limits.max_storage_buffers_per_shader_stage >= 4 && limits.max_compute_workgroups_per_dimension > 0
    }).expect("FATAL: No capable GPU or Software Renderer found!");
    
    println!("[1] Selected Capable Adapter: {:?}", adapter.get_info().name);

    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Organon XLA Device"),
        required_limits: adapter.limits(),
        ..Default::default()
    }).await.expect("Failed to request GPU device");

    // 1. Define the Neural Network Block (PyTorch-style code!)
    // We pass our normal block of code into `organon::lazy::compile`
    println!("\n[2] Compiling Neural Network Block into JIT Cache...");
    
    let compiled_model = compile(&device, |inputs| {
        let a = inputs[0]; // Matrix A
        let b = inputs[1]; // Matrix B
        let bias = inputs[2]; // Bias Vector
        
        let mm = LazyBackend::matmul(a, b);    // <-- Compiler breaks fusion here to run heavy MatMul
        let add = LazyBackend::add(&mm, bias); // <-- Compiler starts new Fused shader here!
        LazyBackend::relu(&add)
    }, &[
        &LazyBackend::new_cpu(vec![], vec![2, 4]), // Dummy A: 2x4
        &LazyBackend::new_cpu(vec![], vec![4, 3]), // Dummy B: 4x3
        &LazyBackend::new_cpu(vec![], vec![1]),    // Dummy Bias
    ]);

    println!("    -> Successfully built Execution Plan! Steps: {}", compiled_model.steps.len());

    // 2. Create raw GPU buffers with real numbers
    use wgpu::util::DeviceExt;
    let create_buf = |data: &[f32], label: &str| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor { 
            label: Some(label), 
            contents: bytemuck::cast_slice(data), 
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC 
        })
    };
    
    // Matrix A: 2x4
    let a_data: [f32; 8] = [1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0];
    // Matrix B: 4x3
    let b_data: [f32; 12] = [1.0, 0.0, 1.0,  0.0, 1.0, 0.0,  1.0, 0.0, 1.0,  0.0, 1.0, 0.0];
    // Bias: Scalar
    let c_data: [f32; 1] = [10.0]; 

    let a_buf = create_buf(&a_data, "Input A");
    let b_buf = create_buf(&b_data, "Input B");
    let c_buf = create_buf(&c_data, "Input C (Bias)");

    // 3. Execute the Cached JIT Model!
    println!("\n[3] Executing Cached Plan on Hardware...");
    let inputs = [&a_buf, &b_buf, &c_buf];
    
    // THIS is the function you will run 500 times in your training loop!
    let out_buffer = compiled_model.execute(&device, &queue, &inputs);

    // 4. Read the results back to the CPU
    println!("\n[4] Reading results from VRAM...");
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging"), 
        size: (2 * 3 * 4) as wgpu::BufferAddress, // Result is 2x3 matrix (2 rows, 3 cols) = 6 floats!
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, 
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&out_buffer, 0, &staging_buf, 0, 2 * 3 * 4);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let results: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    
    println!("----------------------------------------------------");
    println!("Expected Math: ReLU((A * B) + C)");
    println!("Expected Shape: 2x3 Matrix");
    println!("Actual Result: {:?}", results);
    println!("----------------------------------------------------");
}

fn main() {
    pollster::block_on(run_jit_execution());
}