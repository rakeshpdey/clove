// run cargo test after compiling & correct order

pub mod tensor;
pub mod nn;
pub mod optim;
pub mod device;
// pub mod complex_tensor;

pub use device::EngineDevice;
pub use tensor::{Tensor, Node};
pub use optim::SGD;

#[cfg(test)]
mod tests {
    use super::nn::Linear;
    use super::optim::SGD;
    use super::tensor::{Tensor, Node};
    // use super::complex_tensor::ComplexTensor;
    // use num_complex::Complex32;
    use ndarray::{array, Array2};
    use std::sync::Arc; // UPGRADED: Replaced Rc with Arc

    // =========================================================
    // TEST HELPERS: Bridging Arc<RwLock> back to ndarray 
    // =========================================================
    fn to_arr(node: &Node) -> Array2<f32> {
        let t = node.read().unwrap();
        if let crate::tensor::TensorData::Cpu(ref vec) = t.data {
            Array2::from_shape_vec((t.shape[0], t.shape[1]), vec.clone()).expect("Shape mismatch")
        } else {
            panic!("Test Failed: Cannot extract ndarray from GPU Tensor.");
        }
    }

    fn grad_to_arr(node: &Node) -> Array2<f32> {
        let t = node.read().unwrap();
        let grad_slice = t.get_cpu_grad();
        Array2::from_shape_vec((t.shape[0], t.shape[1]), grad_slice.to_vec())
            .expect("Grad shape mismatch")
    }

    // ---------------------------------------------------------
    // TEST 1: The Forward Pass & Memory Graph
    // ---------------------------------------------------------
    #[test]
    fn test_forward_pass_and_graph() {
        println!("\n--- BOOTING CUSTOM TENSOR ENGINE ---");

        let x = Tensor::new(array![[2.0, -1.0]]);
        let w = Tensor::new(array![
            [0.5, 0.1],
            [1.5, -0.8]
        ]);
        let b = Tensor::new(array![[0.1, 0.5]]);
        
        println!("Input, Weights, and Bias loaded into memory.");

        let step1_matmul = Tensor::matmul(&x, &w);
        let step2_add = Tensor::add(&step1_matmul, &b);
        let output_y = Tensor::relu(&step2_add);

        println!("\nFinal Output Matrix after ReLU:\n{}", to_arr(&output_y));

        let has_history = !output_y.read().unwrap().creators.is_empty();
        println!("Did the output tensor remember its parents? {}", has_history);
        
        assert!(has_history, "CRITICAL FAILURE: Tape recorder did not link the graph!");
        
        println!("--- TEST SUCCESSFUL ---\n");
    }

    // ---------------------------------------------------------
    // TEST 2: The Calculus Autograd Engine
    // ---------------------------------------------------------
    #[test]
    fn test_autograd_calculus() {
        println!("\n--- BOOTING AUTOGRAD ENGINE ---");

        let x = Tensor::new(array![[2.0, -1.0]]);
        let w = Tensor::new(array![
            [0.5, 0.1],
            [1.5, -0.8]
        ]);
        let b = Tensor::new(array![[0.1, 0.5]]);
        
        let step1_matmul = Tensor::matmul(&x, &w);
        let step2_add = Tensor::add(&step1_matmul, &b);
        let output_y = Tensor::relu(&step2_add);

        println!("Output Math:\n{}", to_arr(&output_y));

        println!("\nTriggering Chain Rule (.backward())...");
        Tensor::backward(&output_y);

        let w_grad = grad_to_arr(&w);
        println!("\nWeight Gradients (The 'Learning' Signal):\n{}", w_grad);
        
        assert_eq!(w_grad[[0, 0]], 0.0, "Calculus failed: Left column gradient should be zero due to ReLU");
        
        println!("--- AUTOGRAD COMPLETE ---\n");
    }

    // ---------------------------------------------------------
    // TEST 3: The Neural Network Factory (Linear Layer)
    // ---------------------------------------------------------
    #[test]
    fn test_linear_layer() {
        println!("\n--- BOOTING NEURAL NETWORK FACTORY ---");

        let layer = Linear::new(3, 2);
        
        println!("Layer successfully created with random Kaiming weights.");
        println!("Weights:\n{}", to_arr(&layer.weights));
        println!("Bias:\n{}", to_arr(&layer.bias));

        let x = Tensor::new(array![[1.0, 2.0, 3.0]]);
        
        let output = layer.forward(&x);

        println!("\nLayer Output Math:\n{}", to_arr(&output));
        
        let has_history = !output.read().unwrap().creators.is_empty();
        assert!(has_history, "CRITICAL FAILURE: Linear layer broke the tape recorder!");

        println!("--- FACTORY TEST COMPLETE ---\n");
    }

    #[test]
    fn test_training_loop() {
        println!("\n--- BOOTING AI TRAINING LOOP ---");

        let input_x = Tensor::new(array![[1.0, 2.0]]);
        let target_y = array![[0.5]];

        let layer = Linear::new(2, 1);

        let params = vec![Arc::clone(&layer.weights), Arc::clone(&layer.bias)];
        let optimizer = SGD::new(0.05, params);

        println!("Initial random prediction: {}", to_arr(&layer.forward(&input_x)));

        for epoch in 1..=50 {
            let pred = layer.forward(&input_x);
            let loss = Tensor::mse(&pred, &target_y);

            optimizer.zero_grad();
            Tensor::backward(&loss); 
            optimizer.step();

            if epoch == 1 || epoch % 10 == 0 {
                let current_loss = to_arr(&loss)[[0, 0]];
                println!("Epoch {}: Loss (Error) = {:.6}", epoch, current_loss);
            }
        }

        let final_pred = layer.forward(&input_x);
        println!("\nFinal AI Prediction: {}", to_arr(&final_pred));
        
        let prediction_value = to_arr(&final_pred)[[0,0]];
        assert!((prediction_value - 0.5).abs() < 0.05, "The AI failed to learn!");

        println!("--- TRAINING COMPLETE ---\n");
    }

    // ---------------------------------------------------------
    // TEST 5: TRUE DEEP LEARNING (The XOR Problem)
    // ---------------------------------------------------------
    #[test]
    fn test_deep_learning_xor() {
        println!("\n--- BOOTING DEEP LEARNING (XOR PROBLEM) ---");

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

        let layer1 = Linear::new(2, 16);
        let layer2 = Linear::new(16, 16);
        let layer3 = Linear::new(16, 1);

        let mut params = Vec::new();
        params.push(Arc::clone(&layer1.weights)); params.push(Arc::clone(&layer1.bias));
        params.push(Arc::clone(&layer2.weights)); params.push(Arc::clone(&layer2.bias));
        params.push(Arc::clone(&layer3.weights)); params.push(Arc::clone(&layer3.bias));

        let optimizer = SGD::new(0.1, params);

        println!("Training Deep Network for 2,000 Epochs...");

        for epoch in 1..=2000 {
            let out1 = layer1.forward(&x);
            let relu1 = Tensor::relu(&out1);
            
            let out2 = layer2.forward(&relu1);
            let relu2 = Tensor::relu(&out2);
            
            let pred = layer3.forward(&relu2);
            let loss = Tensor::mse(&pred, &y);

            optimizer.zero_grad();
            Tensor::backward(&loss);
            optimizer.step();

            if epoch % 500 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch, to_arr(&loss)[[0,0]]);
            }
        }

        let out1 = layer1.forward(&x);
        let relu1 = Tensor::relu(&out1);
        let out2 = layer2.forward(&relu1);
        let relu2 = Tensor::relu(&out2);
        let final_pred = layer3.forward(&relu2);
        
        println!("\nFinal Deep Learning Predictions for XOR:");
        println!("Expected: [0, 1, 1, 0]");
        println!("Actual AI Output:\n{}", to_arr(&final_pred));
        
        println!("--- DEEP LEARNING COMPLETE ---\n");
    }

    // ---------------------------------------------------------
    // TEST 6: PROBABILITY & STATISTICS (Binary Classification)
    // ---------------------------------------------------------
    #[test]
    fn test_statistical_classification() {
        println!("\n--- BOOTING CLASSIFICATION ENGINE ---");

        let x = Tensor::new(array![
            [10.0, 0.2],
            [0.1,  5.0],
            [12.0, 0.1],
            [0.5,  4.5] 
        ]);
        
        let y = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0] 
        ];

        let layer1 = Linear::new(2, 8);
        let layer2 = Linear::new(8, 2);

        let mut params = Vec::new();
        params.push(Arc::clone(&layer1.weights)); params.push(Arc::clone(&layer1.bias));
        params.push(Arc::clone(&layer2.weights)); params.push(Arc::clone(&layer2.bias));

        let optimizer = SGD::new(0.05, params);

        println!("Training Classifier for 500 Epochs...");

        for epoch in 1..=500 {
            let out1 = layer1.forward(&x);
            let relu1 = Tensor::relu(&out1);
            let raw_logits = layer2.forward(&relu1); 
            
            let loss = Tensor::cross_entropy(&raw_logits, &y);

            optimizer.zero_grad();
            Tensor::backward(&loss);
            optimizer.step();

            if epoch % 100 == 0 {
                println!("Epoch {}: Cross-Entropy Loss = {:.6}", epoch, to_arr(&loss)[[0,0]]);
            }
        }

        let out1 = layer1.forward(&x);
        let relu1 = Tensor::relu(&out1);
        let final_logits = layer2.forward(&relu1);
        
        let probabilities = to_arr(&Tensor::softmax(&final_logits));
        
        println!("\nFinal Statistical Probabilities for Dataset:");
        println!("[Probability Cat A  ,  Probability Cat B]");
        println!("{}", probabilities);

        assert!(probabilities[[1, 1]] > 0.90, "Classifier failed to identify the statistical anomaly!");
        
        println!("--- CLASSIFICATION COMPLETE ---\n");
    }

    // ---------------------------------------------------------
    // TEST 7: ADVANCED LINEAR ALGEBRA (2D Spatial Convolution)
    // ---------------------------------------------------------
    #[test]
    fn test_spatial_convolution() {
        println!("\n--- BOOTING SPATIAL COMPUTER VISION ENGINE ---");

        let image = Tensor::new(array![
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0]
        ]);

        let mut rng = rand::rng();
        let normal = rand_distr::Normal::new(0.0, 0.1).unwrap();
        let kernel_data = Array2::from_shape_fn((3, 3), |_| rand_distr::Distribution::sample(&normal, &mut rng));
        let kernel = Tensor::new(kernel_data);

        let target = array![
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ];

        let params = vec![Arc::clone(&kernel)];
        let optimizer = SGD::new(0.05, params);

        println!("Training Convolutional Kernel for 300 Epochs...");

        for epoch in 1..=300 {
            let conv_output = Tensor::conv2d(&image, &kernel);
            let loss = Tensor::mse(&conv_output, &target);

            optimizer.zero_grad();
            Tensor::backward(&loss);
            optimizer.step();

            if epoch % 50 == 0 {
                println!("Epoch {}: Spatial Loss = {:.6}", epoch, to_arr(&loss)[[0,0]]);
            }
        }

        println!("\nFinal Learned 3x3 Kernel Weights:");
        println!("{}", to_arr(&kernel));
        
        let final_output = Tensor::conv2d(&image, &kernel);
        println!("\nAI's Vision Output (Notice how it highlights the center line):");
        println!("{}", to_arr(&final_output));

        assert!((to_arr(&final_output)[[1, 1]] - 1.0).abs() < 0.1, "Kernel failed to learn spatial edges!");
        
        println!("--- COMPUTER VISION ENGINE COMPLETE ---\n");
    }

    // ---------------------------------------------------------
    // TEST 8: PHYSICS-INFORMED NEURAL NETWORKS (DAG BUG FIXED)
    // ---------------------------------------------------------
    #[test]
    fn test_physics_informed_neural_network() {
        println!("\n--- BOOTING PHYSICS ENGINE (ALGEBRAIC OPTIMIZATION) ---");

        let t_init = Tensor::new(array![[0.0]]);
        let target_init = array![[0.90]];

        let t_physics = Tensor::new(array![
            [0.0], [1.0], [2.0], [3.0], [4.0],
            [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]
        ]);
        
        let t_eps = Tensor::new(array![
            [0.1], [1.1], [2.1], [3.1], [4.1],
            [5.1], [6.1], [7.1], [8.1], [9.1], [10.1]
        ]);

        let minus_point_99 = Tensor::new(array![[-0.99]]);
        let minus_k_env = Tensor::new(array![
            [-0.0022], [-0.0022], [-0.0022], [-0.0022], [-0.0022],
            [-0.0022], [-0.0022], [-0.0022], [-0.0022], [-0.0022], [-0.0022]
        ]);
        
        let zeros = array![
            [0.0], [0.0], [0.0], [0.0], [0.0],
            [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
        ];

        let layer1 = Linear::new(1, 16);
        let layer2 = Linear::new(16, 1);
        
        let mut params = Vec::new();
        params.push(Arc::clone(&layer1.weights)); params.push(Arc::clone(&layer1.bias));
        params.push(Arc::clone(&layer2.weights)); params.push(Arc::clone(&layer2.bias));

        let optimizer = SGD::new(0.01, params);

        println!("Training Physics Engine for 2,000 Epochs...");

        for epoch in 1..=2000 {
            optimizer.zero_grad();

            let out_init = layer1.forward(&t_init);
            let relu_init = Tensor::relu(&out_init);
            let pred_init = layer2.forward(&relu_init);
            let data_loss = Tensor::mse(&pred_init, &target_init);
            Tensor::backward(&data_loss); 

            let out_t = layer1.forward(&t_physics);
            let relu_t = Tensor::relu(&out_t);
            let t_current = layer2.forward(&relu_t);

            let out_eps = layer1.forward(&t_eps);
            let relu_eps = Tensor::relu(&out_eps);
            let t_future = layer2.forward(&relu_eps);

            let scaled_t_current = Tensor::matmul(&t_current, &minus_point_99);
            let step1 = Tensor::add(&t_future, &scaled_t_current);
            let physics_eq = Tensor::add(&step1, &minus_k_env);
            
            let physics_loss = Tensor::mse(&physics_eq, &zeros);
            Tensor::backward(&physics_loss); 

            optimizer.step();

            if epoch % 500 == 0 {
                let dl = to_arr(&data_loss)[[0,0]];
                let pl = to_arr(&physics_loss)[[0,0]];
                println!("Epoch {}: Data Error = {:.6}, Physics Error = {:.6}", epoch, dl, pl);
            }
        }

        let final_out = layer1.forward(&t_physics);
        let final_relu = Tensor::relu(&final_out);
        let final_preds = layer2.forward(&final_relu);
        
        let human_readable_temps = &to_arr(&final_preds) * 100.0;
        
        println!("\nFinal AI Temperature Predictions (Minutes 0 to 10):");
        println!("{}", human_readable_temps);
        
        println!("--- PHYSICS ENGINE COMPLETE ---\n");
    }

    // ---------------------------------------------------------
    // TEST 9: ADVANCED CALCULUS (Higher-Order Derivatives)
    // ---------------------------------------------------------
    #[test]
    fn test_second_order_calculus() {
        println!("\n--- BOOTING HIGHER-ORDER CALCULUS ENGINE ---");

        let x_sgd = Tensor::new(array![[10.0]]);
        let learning_rate = 0.1;
        
        println!("\n[RACE 1] Standard Gradient Descent (First-Order)");
        let mut sgd_steps = 0;
        
        for epoch in 1..=100 {
            x_sgd.write().unwrap().grad = None; 
            
            let y = Tensor::matmul(&x_sgd, &x_sgd);
            Tensor::backward(&y);
            
            let grad = grad_to_arr(&x_sgd);
            
            // Adjust weights via safe inner lock traversal
            if let crate::tensor::TensorData::Cpu(ref mut d) = x_sgd.write().unwrap().data {
                d[0] -= grad[[0,0]] * learning_rate;
            }
            
            sgd_steps = epoch;
            if to_arr(&x_sgd)[[0,0]].abs() < 0.001 { break; } 
        }
        println!("First-Order SGD reached the bottom in {} steps.", sgd_steps);


        println!("\n[RACE 2] Newton's Method (Second-Order Curvature)");
        let x_newton = Tensor::new(array![[10.0]]);
        
        let y1 = Tensor::matmul(&x_newton, &x_newton);
        Tensor::backward(&y1);
        let grad1 = grad_to_arr(&x_newton); 
        
        let epsilon = 0.001;
        let x_eps = Tensor::new(&to_arr(&x_newton) + epsilon);
        
        let y2 = Tensor::matmul(&x_eps, &x_eps);
        Tensor::backward(&y2);
        let grad2 = grad_to_arr(&x_eps); 
        
        let hessian = (&grad2 - &grad1) / epsilon;
        
        if let crate::tensor::TensorData::Cpu(ref mut d) = x_newton.write().unwrap().data {
            d[0] -= grad1[[0,0]] / hessian[[0,0]];
        }
        
        println!("Second-Order Newton reached the bottom in exactly 1 step.");
        println!("Final Coordinate: {:.8}", to_arr(&x_newton)[[0,0]]);
        
        assert!(to_arr(&x_newton)[[0,0]].abs() < 0.01, "Second-Order calculus failed!");
        
        println!("--- HIGHER-ORDER ENGINE COMPLETE ---\n");
    }

    // ---------------------------------------------------------
    // TEST 10: TOPOLOGY & GRAPH THEORY (Graph Neural Networks)
    // ---------------------------------------------------------
    #[test]
    fn test_graph_neural_network() {
        println!("\n--- BOOTING GRAPH NEURAL NETWORK (TOPOLOGY ENGINE) ---");

        let a = Tensor::new(array![
            [0.34, 0.33, 0.00, 0.33],
            [0.33, 0.34, 0.00, 0.33],
            [0.00, 0.00, 1.00, 0.00],
            [0.33, 0.33, 0.00, 0.34],
        ]);

        let x = Tensor::new(array![
            [1.0,  1.0],  
            [1.0,  1.0],  
            [-1.0, -1.0], 
            [0.0,  0.0],  
        ]);

        let y = array![
            [1.0], 
            [1.0], 
            [0.0], 
            [1.0], 
        ];

        let layer1 = Linear::new(2, 8);
        let layer2 = Linear::new(8, 1);

        let mut params = Vec::new();
        params.push(Arc::clone(&layer1.weights)); params.push(Arc::clone(&layer1.bias));
        params.push(Arc::clone(&layer2.weights)); params.push(Arc::clone(&layer2.bias));

        let optimizer = SGD::new(0.05, params);

        println!("Training GNN for 500 Epochs...");

        for epoch in 1..=500 {
            let aggregated_x = Tensor::matmul(&a, &x);
            let out1 = layer1.forward(&aggregated_x);
            let relu1 = Tensor::relu(&out1);

            let aggregated_hidden = Tensor::matmul(&a, &relu1);
            let pred = layer2.forward(&aggregated_hidden);

            let loss = Tensor::mse(&pred, &y);

            optimizer.zero_grad();
            Tensor::backward(&loss); 
            optimizer.step();

            if epoch % 100 == 0 {
                println!("Epoch {}: Topology Loss = {:.6}", epoch, to_arr(&loss)[[0,0]]);
            }
        }

        let final_agg_x = Tensor::matmul(&a, &x);
        let final_out1 = layer1.forward(&final_agg_x);
        let final_relu = Tensor::relu(&final_out1);
        let final_agg_hidden = Tensor::matmul(&a, &final_relu);
        let final_pred = layer2.forward(&final_agg_hidden);

        println!("\nFinal Graph Node Classifications:");
        println!("{}", to_arr(&final_pred));

        assert!(to_arr(&final_pred)[[3, 0]] > 0.8, "GNN failed to classify the blank node!");

        println!("--- GRAPH NEURAL NETWORK COMPLETE ---\n");
    }

    // ---------------------------------------------------------
    // TEST 11: STOCHASTIC CALCULUS (Generative Diffusion Model)
    // ---------------------------------------------------------
    #[test]
    fn test_stochastic_diffusion() {
        println!("\n--- BOOTING GENERATIVE DIFFUSION ENGINE ---");

        let clean_signal = array![[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]];

        let layer1 = Linear::new(8, 32);
        let layer2 = Linear::new(32, 8);

        let mut params = Vec::new();
        params.push(Arc::clone(&layer1.weights)); params.push(Arc::clone(&layer1.bias));
        params.push(Arc::clone(&layer2.weights)); params.push(Arc::clone(&layer2.bias));

        let optimizer = SGD::new(0.01, params);
        let mut rng = rand::rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

        println!("Training AI to map Entropy Gradients for 3,000 Epochs...");

        for epoch in 1..=3000 {
            let noise_data = Array2::from_shape_fn((1, 8), |_| rand_distr::Distribution::sample(&normal, &mut rng));
            
            let noise_intensity = 0.5;
            let corrupted_signal_data = &clean_signal + &(&noise_data * noise_intensity);
            let corrupted_tensor = Tensor::new(corrupted_signal_data);

            let out1 = layer1.forward(&corrupted_tensor);
            let relu1 = Tensor::relu(&out1);
            let predicted_noise = layer2.forward(&relu1);

            let loss = Tensor::mse(&predicted_noise, &noise_data);

            optimizer.zero_grad();
            Tensor::backward(&loss);
            optimizer.step();

            if epoch % 1000 == 0 {
                println!("Epoch {}: Entropy Mapping Error = {:.6}", epoch, to_arr(&loss)[[0,0]]);
            }
        }

        println!("\n--- INITIATING DENOISING SEQUENCE ---");
        
        let pure_static = Array2::from_shape_fn((1, 8), |_| rand_distr::Distribution::sample(&normal, &mut rng));
        let mut current_state = pure_static.clone();

        println!("Time Step 0 (Pure Static): \n{:.2}", current_state);

        let denoising_steps = 5;
        let extraction_rate = 0.5;

        for step in 1..=denoising_steps {
            let state_tensor = Tensor::new(current_state.clone());
            
            let out1 = layer1.forward(&state_tensor);
            let relu1 = Tensor::relu(&out1);
            let predicted_noise = layer2.forward(&relu1);

            current_state = current_state - (to_arr(&predicted_noise) * extraction_rate);
            
            println!("Time Step {} (Denoising): \n{:.2}", step, current_state);
        }

        println!("\nFinal Generated Structure: \n{:.2}", current_state);
        println!("Expected Clean Structure:  \n{:.2}", clean_signal);

        println!("--- DIFFUSION ENGINE COMPLETE ---\n");
    }

    /* TODO: Uncomment and upgrade to Arc<RwLock> for v2.0.0
    #[test]
    fn test_complex_initialization() {
        let data = array![[Complex32::new(1.0, 2.0)]];
        let tensor = ComplexTensor::new(data);

        // Intentionally left as .borrow() assuming ComplexTensor hasn't been upgraded to Arc<RwLock> yet.
        let t = tensor.borrow();
        assert_eq!(t.data[[0, 0]].re, 1.0);
        assert_eq!(t.data[[0, 0]].im, 2.0);
        println!("Complex Tensor initialized successfully!");
    }
    
    #[test]
    fn test_frequency_learning() {
        println!("\n--- BOOTING WAVE FREQUENCY DETECTION TEST ---");
        let mut data = Array2::zeros((1, 8));
        for i in 0..8 {
            let phase = (i as f32) * 0.5;
            data[[0, i]] = Complex32::from_polar(1.0, phase);
        }
        let input = ComplexTensor::new(data);
        let weights = ComplexTensor::new(Array2::from_elem((8, 8), Complex32::new(0.5, 0.5)));
        let output = ComplexTensor::matmul(&input, &weights);

        let result = output.borrow();
        println!("Wave Projection Output (first 3 points):");
        println!("{:?}", result.data.slice(ndarray::s![0, 0..3]));

        assert!(result.data[[0,0]].norm() > 0.0, "Frequency projection failed!");
        println!("--- FREQUENCY DETECTION TEST PASSED ---\n");
    }
    */

    // ---------------------------------------------------------
    // PHASE 1 VERIFICATION: ELEMENT-WISE ARITHMETIC CORE
    // ---------------------------------------------------------
    #[test]
    fn test_phase1_elementwise_arithmetic() {
        println!("\n--- RUNNING BASE ARITHMETIC DIAGNOSTIC ---");

        let a = Tensor::new(array![[5.0, 3.0]]);
        let b = Tensor::new(array![[2.0, 4.0]]);

        let out_sub = Tensor::sub(&a, &b);
        assert_eq!(to_arr(&out_sub)[[0, 0]], 3.0);
        assert_eq!(to_arr(&out_sub)[[0, 1]], -1.0);

        let out_mul = Tensor::mul(&a, &b);
        assert_eq!(to_arr(&out_mul)[[0, 0]], 10.0);
        assert_eq!(to_arr(&out_mul)[[0, 1]], 12.0);

        Tensor::backward(&out_mul);

        let grad_a = grad_to_arr(&a);
        let grad_b = grad_to_arr(&b);

        assert_eq!(grad_a[[0, 0]], 2.0); 
        assert_eq!(grad_a[[0, 1]], 4.0); 
        assert_eq!(grad_b[[0, 0]], 5.0); 
        assert_eq!(grad_b[[0, 1]], 3.0);

        println!("Element-wise Arithmetic and Autograd verified successfully!");
        println!("--- BASE ARITHMETIC COMPLETE ---\n");
    }

    // ---------------------------------------------------------
    // PHASE 3 VERIFICATION: TRIGONOMETRIC WAVEFORMS & DERIVATIVES
    // ---------------------------------------------------------
    #[test]
    fn test_phase3_trigonometry_waves() {
        println!("\n--- RUNNING TRIGONOMETRIC WAVE DIAGNOSTIC ---");

        let x = Tensor::new(array![[0.0]]);

        let out_sin = Tensor::sin(&x);
        let out_cos = Tensor::cos(&x);

        assert!((to_arr(&out_sin)[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((to_arr(&out_cos)[[0, 0]] - 1.0).abs() < 1e-6);

        println!("Forward trigonometric wave projections accurate.");

        Tensor::backward(&out_sin);
        let grad_x_from_sin = grad_to_arr(&x);
        assert!((grad_x_from_sin[[0, 0]] - 1.0).abs() < 1e-6);

        x.write().unwrap().grad = None;

        Tensor::backward(&out_cos);
        let grad_x_from_cos = grad_to_arr(&x);
        assert!((grad_x_from_cos[[0, 0]] - 0.0).abs() < 1e-6);

        println!("Trigonometric wave autograd partial derivatives verified successfully!");
        println!("--- TRIGONOMETRIC ENGINE COMPLETE ---\n");
    }

    // ---------------------------------------------------------
    // PHASE 4 VERIFICATION: GEOMETRY (EMBEDDING & TRANSPOSE)
    // ---------------------------------------------------------
    #[test]
    fn test_phase4_geometry_embeddings() {
        println!("\n--- RUNNING GEOMETRIC EMBEDDING DIAGNOSTIC ---");

        let dictionary_weights = Tensor::new(array![
            [0.1, 0.2],
            [0.9, 0.8],
            [0.5, 0.5]
        ]);

        let sentence_indices = array![[2.0], [0.0]];

        let embedded_sentence = Tensor::embedding(&dictionary_weights, &sentence_indices);
        
        let sentence_data = to_arr(&embedded_sentence);
        assert_eq!(sentence_data[[0, 0]], 0.5); 
        assert_eq!(sentence_data[[1, 0]], 0.1); 

        println!("Embedding lookup successfully mapped categorical tokens to geometric vectors.");

        let transposed_sentence = Tensor::transpose(&embedded_sentence);
        assert_eq!(to_arr(&transposed_sentence).dim(), (2, 2));

        Tensor::backward(&transposed_sentence);

        let dict_gradients = grad_to_arr(&dictionary_weights);

        assert!(dict_gradients[[1, 0]] == 0.0 && dict_gradients[[1, 1]] == 0.0);
        assert!(dict_gradients[[0, 0]] != 0.0);

        println!("Sparse Embedding Autograd verified successfully!");
        println!("--- GEOMETRIC ENGINE COMPLETE ---\n");
    }
}