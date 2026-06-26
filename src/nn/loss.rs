use crate::tensor::{TensorGraph, TensorNode};
use crate::backend::Backend;
use ndarray::Array2;

pub struct CrossEntropyLoss;

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossEntropyLoss {
    pub fn new() -> Self { Self }

    pub fn forward<B: Backend>(&self, logits: &TensorNode<B>, targets: &Array2<f32>) -> TensorNode<B> {
        // Because we moved the complex Math to the Backend trait, 
        // this is now a beautiful 1-liner!
        TensorGraph::<B>::cross_entropy(logits, targets)
    }
}

pub struct MSELoss;

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl MSELoss {
    pub fn new() -> Self { Self }

    pub fn forward<B: Backend>(&self, preds: &TensorNode<B>, targets: &Array2<f32>) -> TensorNode<B> {
        // Same here!
        TensorGraph::<B>::mse(preds, targets)
    }
}