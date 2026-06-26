use organon::data::{load_mnist_images, load_mnist_labels};
use organon::nn::Linear;
use organon::optim::SGD;
use organon::tensor::Tensor;
use ndarray::Array2;
use std::sync::Arc;

// Helper to convert a label (like '5') into a probability array [0,0,0,0,0,1,0,0,0,0]
fn one_hot(label: f32) -> Array2<f32> {
    let mut arr = Array2::<f32>::zeros((1, 10));
    arr[[0, label as usize]] = 1.0;
    arr
}

fn main() {
    println!("==================================================");
    println!("👁️  ORGANON VISION INIT: MNIST DATASET");
    println!("==================================================\n");

    // 1. Boot the GPU
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let gpu_device = Arc::new(device);
    let gpu_queue = Arc::new(queue);
    
    // 2. Load the binary images
    println!("[INFO] Parsing raw binary data...");
    let images = load_mnist_images("data/train-images-idx3-ubyte");
    let labels = load_mnist_labels("data/train-labels-idx1-ubyte");

    // 3. Construct the Neural Network (784 pixels -> 128 hidden -> 10 classes)
    let layer1 = Linear::new(784, 128);
    let layer2 = Linear::new(128, 10);

    let mut params = Vec::new();
    params.push(Arc::clone(&layer1.weights)); 
    params.push(Arc::clone(&layer1.bias));
    params.push(Arc::clone(&layer2.weights)); 
    params.push(Arc::clone(&layer2.bias));

    // We teleport the network weights to the VRAM
    for p in &params {
        p.write().unwrap().to_gpu(Arc::clone(&gpu_device), Arc::clone(&gpu_queue));
    }

    let optimizer = SGD::new(0.01, params);

    println!("\n[TRAINING INITIATED ON GPU]");
    println!("--------------------------------------------------");

    let num_epochs = 1000; // We'll train on the first 1000 images for speed
    let mut running_loss = 0.0;

    for i in 0..num_epochs {
        // Teleport the current image to the GPU
        let x = Tensor::new(images[i].clone());
        x.write().unwrap().to_gpu(Arc::clone(&gpu_device), Arc::clone(&gpu_queue));
        
        let target = one_hot(labels[i]);

        // Forward Pass (VRAM)
        let hidden = layer1.forward(&x);
        let relu = Tensor::relu(&hidden); // Uses CPU fallback for ReLU currently
        let logits = layer2.forward(&relu);
        
        // Calculate Loss
        let loss = Tensor::cross_entropy(&logits, &target);

        // Backward Pass & Optimize (VRAM Matmul Calculus!)
        optimizer.zero_grad();
        Tensor::backward(&loss);
        optimizer.step();

        // Download Loss from VRAM/RAM to monitor
        running_loss += loss.read().unwrap().to_cpu()[[0,0]];

        if (i + 1) % 100 == 0 {
            println!("Processed {:04}/{} Images | Avg Loss: {:.4}", i + 1, num_epochs, running_loss / 100.0);
            running_loss = 0.0;
        }
    }

    println!("\n==================================================");
    println!("✅ ORGANON VISION TRAINING COMPLETE");
    println!("==================================================");
}