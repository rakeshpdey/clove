use organon::data_loader::DataLoader; 
use organon::tensor::{Tensor, TensorData};
use organon::device::EngineDevice;
use organon::AdamW; 
use organon::nn::model::LanguageModel; 
use organon::nn::CrossEntropyLoss;
use std::sync::{Arc, RwLock};

fn one_hot_encode(data: &[f32], vocab_size: usize) -> Vec<f32> {
    let mut result = vec![0.0; data.len() * vocab_size];
    for (i, &val) in data.iter().enumerate() {
        let id = val as usize;
        if id < vocab_size { result[i * vocab_size + id] = 1.0; }
    }
    result
}

fn argmax(data: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &val) in data.iter().enumerate() {
        if val > max_val { max_val = val; max_idx = i; }
    }
    max_idx
}

fn backward_graph(root: &Arc<RwLock<Tensor>>) {
    let mut topo = Vec::new();
    let mut visited = std::collections::HashSet::new();
    
    fn build_topo(node: &Arc<RwLock<Tensor>>, topo: &mut Vec<Arc<RwLock<Tensor>>>, visited: &mut std::collections::HashSet<usize>) {
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
    println!("--- ORGANON: Real Language Training (HuggingFace Tokenizer) ---");
    
    // 1. Train on a LARGER window so it learns RoPE angles 0 through 31!
    let seq_len = 32; 
    let loader = DataLoader::from_file("input.txt", seq_len, 1);
    let vocab_size = loader.tokenizer.vocab_size;
    
    let text = std::fs::read_to_string("input.txt").unwrap_or_else(|_| "Mary had a little lamb, its fleece was white as snow.".to_string());
    let tokens = loader.tokenizer.encode(&text);
    
    if tokens.len() <= seq_len + 1 {
        panic!("Please add more text to input.txt! It needs to be at least {} tokens long.", seq_len + 2);
    }

    let model = LanguageModel::new(vocab_size, 128, 1, 4, 4, 512, 1, 0.0); 
    let mut optimizer = AdamW::new(0.001, model.parameters());
    let criterion = CrossEntropyLoss::new();

    println!("Starting Continuous Sliding-Window Training Loop...");

    let num_epochs = 2000; 
    for epoch in 1..=num_epochs {
        let start_idx = epoch % (tokens.len() - seq_len - 1);
        let x_slice = &tokens[start_idx .. start_idx + seq_len];
        let y_slice = &tokens[start_idx + 1 .. start_idx + seq_len + 1];
        
        let x_vec: Vec<f32> = x_slice.iter().map(|&id| id as f32).collect();
        let y_vec: Vec<f32> = y_slice.iter().map(|&id| id as f32).collect();
        
        let y_one_hot = one_hot_encode(&y_vec, vocab_size);

        let x_node = Arc::new(RwLock::new(Tensor { data: TensorData::Cpu(x_vec), shape: vec![seq_len], grad: None, creators: vec![], device: EngineDevice::Cpu { cores: 1 }, backward: None }));
        let y_node = Arc::new(RwLock::new(Tensor { data: TensorData::Cpu(y_one_hot), shape: vec![seq_len, vocab_size], grad: None, creators: vec![], device: EngineDevice::Cpu { cores: 1 }, backward: None }));
        
        let (logits, aux_loss) = model.forward(&x_node);
        let loss = criterion.forward(&logits, &y_node);
        let base_loss_val = if let TensorData::Cpu(d) = &loss.read().unwrap().data { d[0] } else { 0.0 };
        
        optimizer.zero_grad();
        loss.write().unwrap().grad = Some(TensorData::Cpu(vec![1.0]));
        
        backward_graph(&loss);
        optimizer.step();

        if epoch % 200 == 0 {
            println!("Epoch: {} | Total Loss: {:.6}", epoch, base_loss_val + aux_loss);
        }
    }
    
    println!("\nTraining Complete! Commencing Ultra-Fast Autoregressive Generation with KV-Cache...");
    println!("---------------------------------------------------------");
    
    // 2. Start the prompt with ONLY 16 tokens. This leaves us 16 safe RoPE angles to generate with!
    let prompt_len = 16;
    let mut context_tokens = tokens[0..prompt_len].to_vec();
    print!("{}", loader.tokenizer.decode(&context_tokens));

    // ========================================================
    // PHASE 1: Pre-fill the KV Cache
    // ========================================================
    let prompt_data: Vec<f32> = context_tokens.iter().map(|&id| id as f32).collect();
    let prompt_node = Arc::new(RwLock::new(Tensor {
        data: TensorData::Cpu(prompt_data), shape: vec![prompt_len], grad: None, creators: vec![], device: EngineDevice::Cpu { cores: 1 }, backward: None,
    }));

    let (logits_node, _) = model.forward(&prompt_node);

    let mut next_token_id = {
        let logits_tensor = logits_node.read().unwrap();
        if let TensorData::Cpu(d) = &logits_tensor.data {
            let last_row = &d[(prompt_len - 1) * vocab_size .. prompt_len * vocab_size];
            argmax(last_row) 
        } else { panic!("Expected CPU tensor"); }
    };

    print!("{}", loader.tokenizer.decode(&[next_token_id]));
    context_tokens.push(next_token_id);

    // ========================================================
    // PHASE 2: Fast Generation with Cache (seq_len = 1)
    // We only generate 15 tokens so we stay perfectly inside our 32-token RoPE knowledge limit!
    // ========================================================
    for _ in 0..15 { 
        let x_data = vec![next_token_id as f32];

        let x_node = Arc::new(RwLock::new(Tensor {
            data: TensorData::Cpu(x_data), 
            shape: vec![1], 
            grad: None, creators: vec![], device: EngineDevice::Cpu { cores: 1 }, backward: None,
        }));

        let (logits_node, _) = model.forward(&x_node);

        next_token_id = {
            let logits_tensor = logits_node.read().unwrap();
            if let TensorData::Cpu(d) = &logits_tensor.data {
                argmax(d)
            } else { panic!("Expected CPU tensor"); }
        };

        print!("{}", loader.tokenizer.decode(&[next_token_id]));
        context_tokens.push(next_token_id);
    }
    println!("\n\n---------------------------------------------------------");
}