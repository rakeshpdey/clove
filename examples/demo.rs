use organon::nn::Linear;
use organon::optim::SGD;
use organon::tensor::Tensor;
use ndarray::array;
use std::sync::Arc;

fn main() {
    println!("==================================================");
    println!("BOOTING ORGANON v0.1.0 NEURAL ENGINE");
    println!("==================================================\n");

    // 1. THE DATASET (The XOR Problem)
    // AI must learn that identical inputs = 0, mixed inputs = 1
    let x = Tensor::new(array![
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]);
    let y = array![
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ];

    println!("[INFO] Dataset loaded into memory.");

    // 2. THE NEURAL NETWORK (2 Inputs -> 16 Hidden -> 1 Output)
    let layer1 = Linear::new(2, 16);
    let layer2 = Linear::new(16, 1);

    println!("[INFO] Deep Neural Network constructed.");

    // 3. THE OPTIMIZER
    let mut params = Vec::new();
    params.push(Arc::clone(&layer1.weights)); 
    params.push(Arc::clone(&layer1.bias));
    params.push(Arc::clone(&layer2.weights)); 
    params.push(Arc::clone(&layer2.bias));

    let optimizer = SGD::new(0.1, params);

    println!("\n[TRAINING INITIATED]");
    println!("--------------------------------------------------");

    // 4. THE TRAINING LOOP
    for epoch in 1..=2000 {
        // Forward Pass
        let out1 = layer1.forward(&x);
        let relu1 = Tensor::relu(&out1);
        let pred = layer2.forward(&relu1);
        
        // Calculate Error
        let loss = Tensor::mse(&pred, &y);

        // Backward Pass (Calculus & Tape Recorder)
        optimizer.zero_grad();
        Tensor::backward(&loss);
        optimizer.step();

        // Download Loss from VRAM/RAM to print
        if epoch % 500 == 0 || epoch == 1 {
            let loss_val = loss.read().unwrap().to_cpu()[[0,0]];
            println!("Epoch {:04} | Loss: {:.6}", epoch, loss_val);
        }
    }

    // 5. EVALUATION
    println!("--------------------------------------------------");
    println!("\n[TRAINING COMPLETE] Evaluating AI Intelligence...\n");

    let out1 = layer1.forward(&x);
    let relu1 = Tensor::relu(&out1);
    let final_pred = layer2.forward(&relu1);
    
    // Safely extract the final prediction tensor to the CPU
    let final_matrix = final_pred.read().unwrap().to_cpu();

    println!("Target Answers: [0.0, 1.0, 1.0, 0.0]");
    println!("AI Predictions:");
    for i in 0..4 {
        println!("Scenario {}: {:.4}", i + 1, final_matrix[[i, 0]]);
    }
    
    println!("\n==================================================");
    println!("ORGANON ENGINE SHUTDOWN SEQUENCE COMPLETE");
    println!("==================================================");
}