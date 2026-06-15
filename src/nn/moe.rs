use crate::nn::Linear;
use crate::tensor::{Tensor, Node};

pub struct Expert {
    pub w1: Linear,
    pub w2: Linear,
    pub dropout_rate: f32,
}

impl Expert {
    pub fn new(hidden_size: usize, hidden_dim: usize, dropout_rate: f32) -> Self {
        Self {
            w1: Linear::new(hidden_size, hidden_dim),
            w2: Linear::new(hidden_dim, hidden_size),
            dropout_rate,
        }
    }

    pub fn forward(&self, x: &Node) -> Node {
        let h1 = self.w1.forward(x);
        let gelu = Tensor::gelu(&h1);
        let dropped = Tensor::dropout(&gelu, self.dropout_rate);
        self.w2.forward(&dropped)
    }
}

pub struct MoELayer {
    pub num_experts: usize,
    pub top_k: usize,
    pub router: Linear,
    pub experts: Vec<Expert>,
}

impl MoELayer {
    pub fn new(hidden_size: usize, hidden_dim: usize, num_experts: usize, top_k: usize, dropout_rate: f32) -> Self {
        let mut experts = Vec::new();
        for _ in 0..num_experts {
            experts.push(Expert::new(hidden_size, hidden_dim, dropout_rate));
        }
        
        Self {
            num_experts,
            top_k,
            // The router takes a hidden vector and projects it to a score for each expert
            router: Linear::new(hidden_size, num_experts),
            experts,
        }
    }

    pub fn forward(&self, x: &Node) -> Node {
        // 1. Compute routing scores across all experts
        let routing_logits = self.router.forward(x);
        let routing_weights = Tensor::softmax(&routing_logits);
        
        // =====================================================================
        // THE VRAM BRIDGE: Safely extract probabilities from GPU VRAM to RAM!
        // =====================================================================
        let probs_matrix = routing_weights.read().unwrap().to_cpu();
        
        let mut best_expert_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        
        // Evaluate the first row (assuming a sequence/batch size of 1 for the demo)
        for i in 0..self.num_experts {
            let val = probs_matrix[[0, i]];
            if val > max_val {
                max_val = val;
                best_expert_idx = i;
            }
        }
        
        // 2. Route the input dynamically to the chosen expert
        let expert_output = self.experts[best_expert_idx].forward(x);
        
        // 3. Weight the output by the router's confidence score
        // We use our new `mul_scalar` helper to avoid allocating a tiny new tensor!
        Tensor::mul_scalar(&expert_output, max_val)
    }
}
