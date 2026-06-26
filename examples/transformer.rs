use organon::nn::Linear;
use organon::optim::SGD;
use organon::tensor::{Node, Tensor};
use ndarray::array;
use std::sync::Arc;

// =====================================================================
// 1. SELF-ATTENTION (The core of the Transformer)
// Formula: Softmax(Q * K^T / sqrt(d)) * V
// =====================================================================
pub struct SelfAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub scale: f32,
}

impl SelfAttention {
    pub fn new(embed_dim: usize) -> Self {
        Self {
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            scale: 1.0 / (embed_dim as f32).sqrt(),
        }
    }

    pub fn forward(&self, x: &Node) -> Node {
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let k_t = Tensor::transpose(&k);
        let attention_scores = Tensor::matmul(&q, &k_t);
        let scaled_scores = Tensor::mul_scalar(&attention_scores, self.scale);
        let attention_weights = Tensor::softmax(&scaled_scores);
        
        Tensor::matmul(&attention_weights, &v)
    }
}

// =====================================================================
// 2. MIXTURE OF EXPERTS (The Brain Router)
// =====================================================================
pub struct Expert {
    pub w1: Linear,
    pub w2: Linear,
}

impl Expert {
    pub fn new(hidden_size: usize, hidden_dim: usize) -> Self {
        Self {
            w1: Linear::new(hidden_size, hidden_dim),
            w2: Linear::new(hidden_dim, hidden_size),
        }
    }
    pub fn forward(&self, x: &Node) -> Node {
        let h1 = self.w1.forward(x);
        let relu = Tensor::relu(&h1); // Using ReLU for simplicity in this demo
        self.w2.forward(&relu)
    }
}

pub struct MoELayer {
    pub num_experts: usize,
    pub router: Linear,
    pub experts: Vec<Expert>,
}

impl MoELayer {
    pub fn new(hidden_size: usize, hidden_dim: usize, num_experts: usize) -> Self {
        let mut experts = Vec::new();
        for _ in 0..num_experts { experts.push(Expert::new(hidden_size, hidden_dim)); }
        Self { num_experts, router: Linear::new(hidden_size, num_experts), experts }
    }

    pub fn forward(&self, x: &Node) -> Node {
        let routing_logits = self.router.forward(x);
        let routing_weights = Tensor::softmax(&routing_logits);
        let probs_matrix = routing_weights.read().unwrap().to_cpu();
        
        // For this proof of concept, we route the ENTIRE sequence batch to the best expert 
        // based on the first token's routing preference.
        let mut best_expert_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        
        for i in 0..self.num_experts {
            let val = probs_matrix[[0, i]];
            if val > max_val { max_val = val; best_expert_idx = i; }
        }
        
        let expert_output = self.experts[best_expert_idx].forward(x);
        Tensor::mul_scalar(&expert_output, max_val)
    }
}

// =====================================================================
// 3. THE MOE-TRANSFORMER BLOCK
// =====================================================================
pub struct MoETransformer {
    pub attention: SelfAttention,
    pub moe: MoELayer,
}

impl MoETransformer {
    pub fn new(embed_dim: usize, moe_hidden: usize, num_experts: usize) -> Self {
        Self {
            attention: SelfAttention::new(embed_dim),
            moe: MoELayer::new(embed_dim, moe_hidden, num_experts),
        }
    }

    pub fn forward(&self, x: &Node) -> Node {
        // Normally there are residual connections (x + attention(x)) and LayerNorms here,
        // but we are keeping the computational graph lean for the demo!
        let attn_out = self.attention.forward(x);
        self.moe.forward(&attn_out)
    }
}

// =====================================================================
// MAIN TRAINING LOOP
// =====================================================================
fn main() {
    println!("==================================================");
    println!("🧠 ORGANON LANGUAGE ENGINE: MoE GPT");
    println!("==================================================\n");

    let vocab_size = 5;
    let embed_dim = 16;

    // 1. The Dataset: A simple sequence pattern: 1 -> 2 -> 3 -> 4
    let input_sequence = array![[1.0], [2.0], [3.0], [4.0]]; // Shape: [4, 1]
    
    // The target is the NEXT token in the sequence (shifted by 1)
    // Target for 1 is 2. Target for 2 is 3. Target for 4 loops back to 1.
    let target_one_hot = array![
        [0.0, 0.0, 1.0, 0.0, 0.0], // 2
        [0.0, 0.0, 0.0, 1.0, 0.0], // 3
        [0.0, 0.0, 0.0, 0.0, 1.0], // 4
        [0.0, 1.0, 0.0, 0.0, 0.0]  // 1
    ];

    // 2. Build the Model
    println!("[INFO] Assembling MoE Transformer Architecture...");
    let embedding_weights = Tensor::kaiming_random(vocab_size, embed_dim);
    let transformer = MoETransformer::new(embed_dim, 32, 4); // 4 Experts!
    let lm_head = Linear::new(embed_dim, vocab_size);

    // 3. Collect Parameters for the Optimizer
    let mut params = vec![Arc::clone(&embedding_weights), Arc::clone(&lm_head.weights), Arc::clone(&lm_head.bias)];
    params.push(Arc::clone(&transformer.attention.q_proj.weights)); params.push(Arc::clone(&transformer.attention.q_proj.bias));
    params.push(Arc::clone(&transformer.attention.k_proj.weights)); params.push(Arc::clone(&transformer.attention.k_proj.bias));
    params.push(Arc::clone(&transformer.attention.v_proj.weights)); params.push(Arc::clone(&transformer.attention.v_proj.bias));
    params.push(Arc::clone(&transformer.moe.router.weights));      params.push(Arc::clone(&transformer.moe.router.bias));
    for expert in &transformer.moe.experts {
        params.push(Arc::clone(&expert.w1.weights)); params.push(Arc::clone(&expert.w1.bias));
        params.push(Arc::clone(&expert.w2.weights)); params.push(Arc::clone(&expert.w2.bias));
    }

    let mut optimizer = AdamW::new(0.001, model.parameters());

    println!("\n[TRAINING GPT ON MULTI-CORE CPU]");
    println!("--------------------------------------------------");

    let num_epochs = 500;
    for epoch in 1..=num_epochs {
        // Forward Pass: Tokens -> Embeddings -> Attention -> MoE -> Logits
        let embeddings = Tensor::embedding(&embedding_weights, &input_sequence);
        let hidden = transformer.forward(&embeddings);
        let logits = lm_head.forward(&hidden);

        // Loss
        let loss = Tensor::cross_entropy(&logits, &target_one_hot);

        // Backward Pass & Optimize
        optimizer.zero_grad();
        Tensor::backward(&loss);
        optimizer.step();

        if epoch % 50 == 0 || epoch == 1 {
            let loss_val = loss.read().unwrap().to_cpu()[[0,0]];
            println!("Epoch {:03} | Cross-Entropy Loss: {:.4}", epoch, loss_val);
        }
    }

    println!("\n==================================================");
    println!("✅ MoE GPT TRAINING COMPLETE");
    println!("==================================================");
}