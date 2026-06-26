use crate::nn::Linear;
use crate::tensor::{TensorGraph, TensorNode};
use crate::backend::Backend;

pub struct ODEFunc<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> ODEFunc<B> {
    pub fn new(dim: usize) -> Self {
        Self {
            fc1: Linear::new(dim, dim),
            fc2: Linear::new(dim, dim),
        }
    }

    pub fn forward(&self, h: &TensorNode<B>) -> TensorNode<B> {
        let h1 = self.fc1.forward(h);
        let act = TensorGraph::<B>::gelu(&h1);
        self.fc2.forward(&act)
    }

    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }
}

pub struct NeuralODE<B: Backend> {
    pub func: ODEFunc<B>,
    pub num_steps: usize,
    pub dt: f32,
}

impl<B: Backend> NeuralODE<B> {
    pub fn new(dim: usize, num_steps: usize) -> Self {
        Self {
            func: ODEFunc::new(dim),
            num_steps,
            dt: 1.0 / (num_steps as f32),
        }
    }

    pub fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        let mut h = x.clone(); 
        for _ in 0..self.num_steps {
            let dh = self.func.forward(&h);
            let step = TensorGraph::<B>::mul_scalar(&dh, self.dt);
            h = TensorGraph::<B>::add(&h, &step);
        }
        h 
    }

    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        self.func.parameters()
    }
}