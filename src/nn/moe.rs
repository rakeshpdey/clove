use crate::nn::Linear;
use crate::tensor::{TensorGraph, TensorNode};
use crate::backend::Backend;

pub struct Expert<B: Backend> {
    pub w1: Linear<B>,
    pub w2: Linear<B>,
    pub dropout_rate: f32,
}

impl<B: Backend> Expert<B> {
    pub fn new(hidden_size: usize, hidden_dim: usize, dropout_rate: f32) -> Self {
        Self {
            w1: Linear::new(hidden_size, hidden_dim),
            w2: Linear::new(hidden_dim, hidden_size),
            dropout_rate,
        }
    }

    pub fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        let h1 = self.w1.forward(x);
        let gelu = TensorGraph::<B>::gelu(&h1);
        let dropped = TensorGraph::<B>::dropout(&gelu, self.dropout_rate);
        self.w2.forward(&dropped)
    }
    
    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        let mut params = self.w1.parameters();
        params.extend(self.w2.parameters());
        params
    }
}

pub struct MoELayer<B: Backend> {
    pub num_experts: usize,
    pub top_k: usize,
    pub router: Linear<B>,
    pub experts: Vec<Expert<B>>,
}

impl<B: Backend> MoELayer<B> {
    pub fn new(hidden_size: usize, hidden_dim: usize, num_experts: usize, top_k: usize, dropout_rate: f32) -> Self {
        let mut experts = Vec::new();
        for _ in 0..num_experts {
            experts.push(Expert::new(hidden_size, hidden_dim, dropout_rate));
        }
        
        Self {
            num_experts,
            top_k,
            router: Linear::new(hidden_size, num_experts),
            experts,
        }
    }

    pub fn forward(&self, x: &TensorNode<B>) -> (TensorNode<B>, f32) {
        let routing_logits = self.router.forward(x);
        let routing_weights = TensorGraph::<B>::softmax(&routing_logits);
        
        // =====================================================================
        // WARNING: THE XLA TRAP!
        // =====================================================================
        // This `to_cpu()` call forces a sync with physical memory.
        // If you use LazyBackend (XLA), this will panic because the graph hasn't executed yet!
        // To make MoE XLA-compatible, you would need to implement a WGSL TopK shader 
        // so the routing happens entirely on the GPU without CPU intervention.
        let probs_matrix = routing_weights.read().unwrap().to_cpu();

        let mut avg_probs = vec![0.0; self.num_experts];
        let batch_size = probs_matrix.shape()[0] as f32;

        for i in 0..self.num_experts {
            let mut sum = 0.0;
            for b in 0..probs_matrix.shape()[0] {
                sum += probs_matrix[[b, i]];
            }
            avg_probs[i] = sum / batch_size;
        }
        
        let target_prob = 1.0 / self.num_experts as f32;
        let penalty: f32 = avg_probs.iter()
            .map(|&p| (p - target_prob).powi(2))
            .sum::<f32>() * 0.1;
        
        let mut best_expert_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        
        for i in 0..self.num_experts {
            let val = probs_matrix[[0, i]];
            if val > max_val {
                max_val = val;
                best_expert_idx = i;
            }
        }
        
        let expert_output = self.experts[best_expert_idx].forward(x);
        (TensorGraph::<B>::mul_scalar(&expert_output, max_val), penalty)
    }
}