use crate::tensor::{TensorGraph, TensorNode};
use crate::backend::Backend;
use ndarray::Array2;

pub struct Embedding<B: Backend> {
    pub weights: TensorNode<B>,
    pub vocab_size: usize,
    pub hidden_dim: usize,
}

impl<B: Backend> Embedding<B> {
    pub fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        Self {
            weights: TensorGraph::<B>::kaiming_random(vocab_size, hidden_dim),
            vocab_size,
            hidden_dim,
        }
    }

    pub fn forward(&self, indices: &Array2<f32>) -> TensorNode<B> {
        TensorGraph::<B>::embedding(&self.weights, indices)
    }

    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        vec![self.weights.clone()]
    }
}