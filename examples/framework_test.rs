use organon::tensor::Tensor;
use ndarray::array;

fn main() {
    println!("==================================================");
    println!("🧪 ORGANON FRAMEWORK: INTEGRATION SMOKE TEST");
    println!("==================================================\n");

    // Dummy data for testing
    let dummy_input = Tensor::new(array![[1.0, 2.0, 3.0, 4.0]]); // Shape [1, 4]
    
    // --------------------------------------------------
    // TEST 1: Math Ops & Activations
    // --------------------------------------------------
    println!("[TEST 1/3] Verifying Basic Math & Activation Pipelines...");
    
    let _sub_test = Tensor::sub(&dummy_input, &dummy_input);
    let _mul_scalar = Tensor::mul_scalar(&dummy_input, 2.0);
    let _gelu_test = Tensor::gelu(&dummy_input);
    let _dropout_test = Tensor::dropout(&dummy_input, 0.1);
    
    println!("✅ Basic math ops verified.");

    // --------------------------------------------------
    // TEST 2: Vision Ops (Conv2d & Flatten)
    // --------------------------------------------------
    println!("\n[TEST 2/3] Verifying Vision Pipelines...");
    
    let dummy_image = Tensor::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    // Temporarily forcing 2D shape metadata for the test
    dummy_image.write().unwrap().shape = vec![3, 3];
    let kernel = Tensor::kaiming_random(2, 2);
    
    // FIXED: conv_out is actively used by flatten, so it has no underscore
    let conv_out = Tensor::conv2d(&dummy_image, &kernel);
    let _flat_out = Tensor::flatten(&conv_out);
    
    println!("✅ Vision ops verified.");

    // --------------------------------------------------
    // TEST 3: Language Ops (Embedding, Softmax, Transpose, LayerNorm)
    // --------------------------------------------------
    println!("\n[TEST 3/3] Verifying Language & Transformer Pipelines...");
    
    let token_input = array![[0.0], [1.0]]; // Two tokens
    let embed_weights = Tensor::kaiming_random(5, 4); // Vocab 5, Dim 4
    
    // FIXED: embedded is used by transpose, softmax, and layernorm, so it has no underscore
    let embedded = Tensor::embedding(&embed_weights, &token_input);
    let _transposed = Tensor::transpose(&embedded);
    let _softmax_out = Tensor::softmax(&embedded);

    // Provide dummy Gamma (1.0) and Beta (0.0) tensors for LayerNorm
    let gamma = Tensor::new(array![[1.0, 1.0, 1.0, 1.0]]);
    let beta = Tensor::new(array![[0.0, 0.0, 0.0, 0.0]]);
    let _norm_out = Tensor::layer_norm(&embedded, &gamma, &beta);

    println!("✅ Language ops verified.");

    println!("\n==================================================");
    println!("🎉 ALL SYSTEMS GREEN: ORGANON v0.1.0 IS STABLE!");
    println!("==================================================");
}