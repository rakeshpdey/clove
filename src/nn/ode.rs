/*
 * src/nn/ode.rs
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Rakesh Pradip Dey
 *
 * Licensed under the MIT License <LICENSE-MIT or http://opensource.org/licenses/MIT>.
 *
 * Note: Portions of this software are adapted from existing open-source frameworks.
 * This file may not be copied, modified, or distributed except according to the terms
 * of the MIT license.
 */
//! Neural Ordinary Differential Equations (ODE) for the Organon Engine.
//!
//! This module implements a Neural ODE layer, where the hidden state dynamics
//! are modeled as a continuous function. Instead of traditional discrete
//! layers, this module approximates the solution to $dh/dt = f(h, t)$ using
//! numerical integration (Euler method).

use crate::backend::Backend;
use crate::nn::Linear;
use crate::tensor::{TensorGraph, TensorNode};

/// Represents the derivative function $f(h, t)$ for the Neural ODE.
///
/// This component acts as the "velocity" model that learns the hidden state dynamics.
pub struct ODEFunc<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> ODEFunc<B> {
    /// Creates a new ODE Function component with the specified dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            fc1: Linear::new(dim, dim),
            fc2: Linear::new(dim, dim),
        }
    }

    /// Performs the forward pass to compute the derivative.
    ///
    /// # Arguments
    /// * `h` - The current hidden state tensor.
    pub fn forward(&self, h: &TensorNode<B>) -> TensorNode<B> {
        let h1 = self.fc1.forward(h);
        let act = TensorGraph::<B>::gelu(&h1);
        self.fc2.forward(&act)
    }

    /// Returns the flattened list of trainable parameters for the derivative function.
    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }
}

/// A Neural Ordinary Differential Equation (ODE) layer.
///
/// This layer integrates the dynamics defined by the `ODEFunc` over a set number of steps,
/// approximating the solution to an ODE $dh/dt = f(h, t)$.
pub struct NeuralODE<B: Backend> {
    pub func: ODEFunc<B>,
    pub num_steps: usize,
    pub dt: f32,
}

impl<B: Backend> NeuralODE<B> {
    /// Creates a new Neural ODE instance.
    ///
    /// # Arguments
    /// * `dim` - The dimensionality of the state.
    /// * `num_steps` - The number of integration steps for the Euler method.
    pub fn new(dim: usize, num_steps: usize) -> Self {
        Self {
            func: ODEFunc::new(dim),
            num_steps,
            dt: 1.0 / (num_steps as f32),
        }
    }

    /// Performs the forward integration pass using the Euler method.
    ///
    /// # Arguments
    /// * `x` - The initial state tensor.
    pub fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        let mut h = x.clone();
        for _ in 0..self.num_steps {
            // Compute the gradient of the hidden state
            let dh = self.func.forward(&h);
            // Apply step: h_next = h + dh * dt
            let step = TensorGraph::<B>::mul_scalar(&dh, self.dt);
            h = TensorGraph::<B>::add(&h, &step);
        }
        h
    }

    /// Returns the trainable parameters for the Neural ODE.
    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        self.func.parameters()
    }
}

impl<B: Backend> crate::nn::Module<B> for NeuralODE<B> {
    fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<TensorNode<B>> {
        self.parameters()
    }
}
