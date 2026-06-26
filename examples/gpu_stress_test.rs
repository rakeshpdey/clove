use organon::tensor::Tensor;
use ndarray::array;
use std::sync::Arc;

fn main() {
    println!("==================================================");
    println!("🔥 BOOTING PURE GPU STRESS TEST");
    println!("==================================================\n");

    // 1. Ignite the Graphics Card
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
    
    // THE FIX: wgpu v29 only takes 1 argument for request_device
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    
    let gpu_device = Arc::new(device);
    let gpu_queue = Arc::new(queue);
    
    println!("[INFO] Graphics Card Online: {:?}", adapter.get_info().name);

    // 2. Create standard CPU matrices
    let w = Tensor::new(array![
        [2.0, 3.0], 
        [1.0, 4.0]
    ]);
    let x = Tensor::new(array![
        [1.0], 
        [2.0]
    ]);

    // 3. Teleport them to VRAM!
    w.write().unwrap().to_gpu(Arc::clone(&gpu_device), Arc::clone(&gpu_queue));
    x.write().unwrap().to_gpu(Arc::clone(&gpu_device), Arc::clone(&gpu_queue));
    println!("[INFO] Matrices teleported to VRAM.");

    // 4. Pure GPU Matrix Multiplication
    println!("[INFO] Executing WGPU MatMul Forward Pass...");
    let y = Tensor::matmul(&w, &x);

    // 5. Pure GPU Backward Calculus
    println!("[INFO] Executing WGPU MatMul Backward Calculus...");
    Tensor::backward(&y);

    // 6. Download the gradients from the GPU to verify!
    println!("\n[SUCCESS] Downloading computed gradients...");
    let w_grad = w.read().unwrap().grad_to_cpu().unwrap();
    let x_grad = x.read().unwrap().grad_to_cpu().unwrap();

    println!("\n∇W (Should be [[1, 2], [1, 2]]):");
    println!("{}", w_grad);
    
    println!("\n∇x (Should be [[3], [7]]):");
    println!("{}", x_grad);
}