use crate::tensor::{TensorGraph, TensorNode};
use crate::backend::Backend;
use std::marker::PhantomData;

pub struct MultiHeadAttention<B: Backend> {
    pub num_heads: usize,
    pub hidden_size: usize,
    pub head_dim: usize,
    _marker: PhantomData<B>,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        Self {
            num_heads, 
            hidden_size, 
            head_dim: hidden_size / num_heads,
            _marker: PhantomData,
        }
    }

    pub fn forward(&self, q: &TensorNode<B>, k: &TensorNode<B>, v: &TensorNode<B>) -> TensorNode<B> {
        // Look how beautiful this is! No WGSL strings, no buffer mapping, no device queues.
        // We write the pure mathematical equation for Attention.
        // The XLA Pattern Matcher will automatically detect `MatMul(Softmax(MatMul))` 
        // and replace this entire function with a FlashAttention Mega-Kernel!
        
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let q_scaled = TensorGraph::<B>::mul_scalar(q, scale);
        
        let qk = TensorGraph::<B>::matmul(&q_scaled, k);
        let scores = TensorGraph::<B>::softmax(&qk);
        TensorGraph::<B>::matmul(&scores, v)
    }
}