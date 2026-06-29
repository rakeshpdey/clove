use clove::nn::Linear;
use clove::optim::AdamW;
use clove::tensor::Tensor;
use ndarray::array;
use std::sync::Arc;

/// This is the "Hello World" of the Clove Framework.
/// It demonstrates how to create a simple Neural Network layer,
/// pass data through it, and train it using the AdamW optimizer and Autograd.
fn main() {
    println!("Booting Clove Framework...\n");

    // 1. DEFINE THE DATA
    let input_x = Tensor::new(array![[1.0, 2.0]]);
    let target_y = array![[0.5]];

    // 2. BUILD THE MODEL
    let layer = Linear::new(2, 1);

    // 3. SETUP THE OPTIMIZER
    let params = vec![Arc::clone(&layer.weights), Arc::clone(&layer.bias)];
    let mut optimizer = AdamW::new(0.05, params);

    println!("Training Model for 100 Epochs...");

    // 4. THE TRAINING LOOP
    for epoch in 1..=100 {
        let prediction = layer.forward(&input_x);
        let loss = Tensor::mse(&prediction, &target_y);

        optimizer.zero_grad();
        Tensor::backward(&loss);
        optimizer.step();

        if epoch % 25 == 0 {
            let loss_val = loss.read().unwrap().to_cpu()[[0, 0]];
            println!("Epoch {:03} | Loss: {:.6}", epoch, loss_val);
        }
    }

    // 5. EVALUATE
    let final_prediction = layer.forward(&input_x).read().unwrap().to_cpu()[[0, 0]];
    println!("\nTraining Complete!");
    println!("Target Value: 0.5");
    println!("AI Predicted: {:.6}", final_prediction);
}