pub mod moe;
pub mod attention;
pub use moe::{MoELayer, Expert};
pub use attention::MultiHeadAttention;
use crate::tensor::{Tensor, Node};
use ndarray::Array2;

// 1. STANDARD LINEAR LAYER (With Bias)
pub struct Linear {
    pub weights: Node,
    pub bias: Node,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let b_data = Array2::zeros((1, output_size));

        Linear {
            weights: Tensor::kaiming_random(input_size, output_size),
            bias: Tensor::new(b_data),
        }
    }
    pub fn forward(&self, input: &Node) -> Node {
        let matmul_result = Tensor::matmul(input, &self.weights);
        Tensor::add(&matmul_result, &self.bias)
    }
}

// 2. LAYER NORMALIZATION
pub struct LayerNorm {
    pub gamma: Node,
    pub beta: Node,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            // DO NOT CHANGE: LayerNorm mathematically requires 1s and 0s
            gamma: Tensor::new(Array2::ones((1, dim))),
            beta: Tensor::new(Array2::zeros((1, dim))),
        }
    }

    pub fn forward(&self, input: &Node) -> Node {
        Tensor::layer_norm(input, &self.gamma, &self.beta)
    }
}

// ========================================================================
// 3. LINEAR LAYER (WITHOUT Bias)
// ========================================================================
// Preserved: The engine does not use bias for Query, Key, Value, or Output projections.
pub struct LinearNoBias {
    pub weights: Node,
}

impl LinearNoBias {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weights: Tensor::kaiming_random(in_features, out_features),
        }
    }

    pub fn forward(&self, input: &Node) -> Node {
        Tensor::matmul(input, &self.weights)
    }
}

// ========================================================================
// 4. THE COMPLETE TRANSFORMER BLOCK (Upgraded for MoE Swarm)
// ========================================================================
// This struct holds the exact parameters in the exact order the engine expects.
pub struct TransformerBlock {
    pub norm1: LayerNorm,
    pub wq: LinearNoBias, 
    pub wk: LinearNoBias,
    pub wv: LinearNoBias,
    pub wo: LinearNoBias,
    pub mha: MultiHeadAttention,
    pub norm2: LayerNorm,
    pub moe: MoELayer,
}

impl TransformerBlock {
    // Look closely at this line! This is what was missing the variables.
    pub fn new(
        hidden_size: usize, 
        hidden_dim: usize, 
        num_experts: usize, 
        num_heads: usize, 
        top_k: usize, 
        dropout_rate: f32
    ) -> Self {
        Self {
            norm1: LayerNorm::new(hidden_size),
            wq: LinearNoBias::new(hidden_size, hidden_size),
            wk: LinearNoBias::new(hidden_size, hidden_size),
            wv: LinearNoBias::new(hidden_size, hidden_size),
            wo: LinearNoBias::new(hidden_size, hidden_size),
            mha: MultiHeadAttention::new(hidden_size, num_heads),
            norm2: LayerNorm::new(hidden_size),
            moe: MoELayer::new(hidden_size, hidden_dim, num_experts, top_k, dropout_rate),
        }
    }
}
