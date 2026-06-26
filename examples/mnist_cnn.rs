use organon::data::{load_mnist_images, load_mnist_labels};
use organon::nn::Linear;
use organon::optim::SGD;
use organon::tensor::Tensor;
use ndarray::Array2;
use std::sync::Arc;

fn one_hot(label: f32) -> Array2<f32> {
    let mut arr = Array2::<f32>::zeros((1, 10));
    arr[[0, label as usize]] = 1.0;
    arr
}

fn main() {
    println!("==================================================");
    println!("👁️  ORGANON VISION: CONVOLUTIONAL NETWORK (CNN)");
    println!("==================================================\n");

    println!("[INFO] Parsing raw binary data...");
    let images = load_mnist_images("data/train-images-idx3-ubyte");
    let labels = load_mnist_labels("data/train-labels-idx1-ubyte");

    // 1. The Computer Vision Architecture
    // A 3x3 kernel to detect edges and shapes
    let conv_kernel = Tensor::kaiming_random(3, 3);
    
    // The image goes from 28x28 -> Conv2d -> 26x26. 
    // 26 * 26 = 676 flattened pixels feeding into 10 output classes.
    let classifier = Linear::new(676, 10);

    let mut params = Vec::new();
    params.push(Arc::clone(&conv_kernel)); 
    params.push(Arc::clone(&classifier.weights)); 
    params.push(Arc::clone(&classifier.bias));

    let optimizer = SGD::new(0.01, params);

    println!("\n[TRAINING CNN ON MULTI-CORE CPU]");
    println!("--------------------------------------------------");

    let num_epochs = 1000; 
    let mut running_loss = 0.0;

    for i in 0..num_epochs {
        // Load the image and reshape it from [1, 784] back to its true [28, 28] 2D grid!
        let x = Tensor::new(images[i].clone());
        x.write().unwrap().shape = vec![28, 28];
        
        let target = one_hot(labels[i]);

        // Forward Pass: The "Vision" Pipeline
        let features = Tensor::conv2d(&x, &conv_kernel);
        let relu = Tensor::relu(&features); 
        let flat = Tensor::flatten(&relu);
        let logits = classifier.forward(&flat);
        
        // Calculate Loss
        let loss = Tensor::cross_entropy(&logits, &target);

        // Backward Pass
        optimizer.zero_grad();
        Tensor::backward(&loss);
        optimizer.step();

        running_loss += loss.read().unwrap().to_cpu()[[0,0]];

        if (i + 1) % 100 == 0 {
            println!("Processed {:04}/{} Images | Avg Loss: {:.4}", i + 1, num_epochs, running_loss / 100.0);
            running_loss = 0.0;
        }
    }

    println!("\n==================================================");
    println!("✅ CNN TRAINING COMPLETE");
    println!("==================================================");
}