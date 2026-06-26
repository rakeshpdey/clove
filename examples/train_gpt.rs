use organon::data::DataLoader;
use organon::tensor::{Tensor, TensorData, Node};
use organon::device::EngineDevice;
use organon::AdamW;
use organon::nn::{LanguageModel, CrossEntropyLoss};
use organon::backend::WgpuBackend;
use std::sync::Arc;

// =====================================================================
// THE AUTOGRAD ENGINE
// Maps the entire neural network backwards to apply gradients!
// =====================================================================
fn backward_graph(root: &Node) {
    let mut topo = Vec::new();
    let mut visited = std::collections::HashSet::new();
    
    fn build_topo(node: &Node, topo: &mut Vec<Node>, visited: &mut std::collections::HashSet<usize>) {
        let ptr = Arc::as_ptr(node) as usize;
        if !visited.contains(&ptr) {
            visited.insert(ptr);
            for child in &node.read().unwrap().creators {
                build_topo(child, topo, visited);
            }
            topo.push(Arc::clone(node));
        }
    }
    
    build_topo(root, &mut topo, &mut visited);
    
    for node in topo.into_iter().rev() {
        let tensor = node.read().unwrap();
        if let Some(back_fn) = &tensor.backward {
            back_fn(&tensor);
        }
    }
}

fn main() {
    println!("====================================================");
    println!("      ORGANON v0.3.0: GENERIC GPT TRAINING          ");
    println!("====================================================\n");

    // 1. Initialize the Fuel Pump (Sequence Length 16, Batch Size 4)
    let seq_len = 16; 
    let batch_size = 4;
    let mut loader = DataLoader::from_file("input.txt", seq_len, batch_size);
    let vocab_size = loader.tokenizer.vocab_size;
    println!("[1] Loaded text! Unique tokens (Vocab Size): {}", vocab_size);

    // 2. Initialize the generic Neural Network
    // We instantiate the Eager Model (WgpuBackend) so AdamW can physically manipulate it!
    // To trace it with XLA, we would instantiate LanguageModel::<LazyBackend>::new(...)
    let model = LanguageModel::<WgpuBackend>::new(vocab_size, 64, 2, 4);
    let criterion = CrossEntropyLoss::new();
    
    // 3. Initialize the AdamW Optimizer
    let mut optimizer = AdamW::new(0.001, model.parameters());

    println!("[2] Neural Network Architecture Compiled & Initialized.");
    println!("[3] Starting Training Loop...");
    
    // 4. The Training Loop
    for epoch in 1..=200 {
        // Fetch a batch of text from the hard drive
        let (x_data, y_data) = loader.next_batch();
        
        // --- FORWARD PASS ---
        // Uses our beautifully clean Generic NN architecture!
        // We pass the raw indices (x_data) directly into the embedding layer.
        let logits = model.forward(&x_data);
        
        // --- LOSS CALCULATION ---
        let loss = criterion.forward::<WgpuBackend>(&logits, &y_data);
        
        // --- BACKWARD PASS ---
        optimizer.zero_grad();
        loss.write().unwrap().grad = Some(TensorData::Cpu(vec![1.0]));
        backward_graph(&loss);
        
        // --- OPTIMIZER STEP (AdamW magic!) ---
        optimizer.step();

        // Print progress
        if epoch % 20 == 0 {
            let loss_val = {
                let l_read = loss.read().unwrap();
                if let TensorData::Cpu(d) = &l_read.data { d[0] } else { 0.0 }
            };
            println!("Epoch: {:03} | Training Loss: {:.4}", epoch, loss_val);
        }
    }

    println!("\n[4] Training Complete!");
    
    // 5. Test the Serialization logic we just restored!
    println!("[5] Saving Model Checkpoint...");
    model.save_weights("organon_gpt_v3.bin").expect("Failed to save weights!");
    println!("====================================================");
}