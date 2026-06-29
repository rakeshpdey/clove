/*
 * src/nn.rs
 * Neural Network Modules
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

//! Neural Network Modules for the Clove Engine.
//!
//! This module serves as the primary API for high-level neural network components.
//! It defines the core `Module` trait and provides a library of pre-built layers,
//! ranging from basic `Linear` layers and normalization (`RMSNorm`) to advanced
//! architectures like `TransformerBlock` and `PagedAttention`.
//!
//! # Architecture
//! The framework follows a modular, trait-based approach. All layers are generic
//! over a `Backend`, ensuring that the same high-level model definition can execute
//! on CPUs, GPUs, or simulated environments simply by changing the backend type.

pub mod attention;
pub mod embedding;
pub mod loss;
pub mod model;
pub mod moe;
pub mod ode;

pub use attention::MultiHeadAttention;
pub use embedding::Embedding;
pub use loss::{CrossEntropyLoss, MSELoss};
pub use model::LanguageModel;
pub use moe::{Expert, MoELayer};
pub use ode::NeuralODE;

use crate::backend::Backend;
use crate::tensor::{TensorGraph, TensorNode};

// THE MODULE TRAIT & SEQUENTIAL API

/// The fundamental trait for all neural network layers.
/// Layers implementing `Module` can be composed into `Sequential` containers
/// and participate in the automated parameter collection for optimizers.
pub trait Module<B: Backend>: Send + Sync {
    /// Perform the forward pass computation.
    fn forward(&self, x: &TensorNode<B>) -> TensorNode<B>;
    /// Return a flattened list of all trainable parameters in the module.
    fn parameters(&self) -> Vec<TensorNode<B>>;
}

/// A container for stacking multiple `Module` layers.
/// Forward passes are executed in the order the layers were added.
pub struct Sequential<B: Backend> {
    layers: Vec<Box<dyn Module<B>>>,
}

impl<B: Backend> Default for Sequential<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Sequential<B> {
    /// Creates an empty Sequential container.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Appends a layer to the container.
    pub fn push<M: Module<B> + 'static>(&mut self, layer: M) {
        self.layers.push(Box::new(layer));
    }

    // Builder pattern convention: takes ownership and allows chaining
    pub fn with_layer<M: Module<B> + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }
}

impl<B: Backend> Module<B> for Sequential<B> {
    fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        let mut current = x.clone();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }
    fn parameters(&self) -> Vec<TensorNode<B>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

// GENERIC LINEAR LAYER

/// A basic affine transformation layer: y = xW^T + b
pub struct Linear<B: Backend> {
    pub weights: TensorNode<B>,
    pub bias: TensorNode<B>,
}

impl<B: Backend> Linear<B> {
    pub fn new(in_feat: usize, out_feat: usize) -> Self {
        Self {
            weights: TensorGraph::<B>::kaiming_random(in_feat, out_feat),
            bias: TensorGraph::<B>::kaiming_random(1, out_feat),
        }
    }

    pub fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        let mm = TensorGraph::<B>::matmul(x, &self.weights);
        TensorGraph::<B>::add(&mm, &self.bias)
    }

    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}

impl<B: Backend> Module<B> for Linear<B> {
    fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<TensorNode<B>> {
        self.parameters()
    }
}

impl<B: Backend> Module<B> for FeedForward<B> {
    fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<TensorNode<B>> {
        self.parameters()
    }
}

impl<B: Backend> Module<B> for TransformerBlock<B> {
    fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<TensorNode<B>> {
        self.parameters()
    }
}

// 2. GENERIC RMS NORMALIZATION

/// Root Mean Square Normalization.
/// Unlike standard BatchNorm, RMSNorm does not center inputs,
/// which improves computational efficiency in Transformer architectures.
pub struct RMSNorm<B: Backend> {
    pub gamma: TensorNode<B>,
    pub beta: TensorNode<B>,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: TensorGraph::<B>::new_cpu(vec![1.0; dim], vec![1, dim]),
            beta: TensorGraph::<B>::new_cpu(vec![0.0; dim], vec![1, dim]),
        }
    }

    pub fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        TensorGraph::<B>::layer_norm(x, &self.gamma, &self.beta)
    }

    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

impl<B: Backend> Module<B> for RMSNorm<B> {
    fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        self.forward(x)
    }
    fn parameters(&self) -> Vec<TensorNode<B>> {
        self.parameters()
    }
}

// 3. GENERIC FEED FORWARD NETWORK

/// A position-wise feed-forward network
pub struct FeedForward<B: Backend> {
    pub w1: Linear<B>,
    pub w2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(dim: usize, hidden_dim: usize) -> Self {
        Self {
            w1: Linear::new(dim, hidden_dim),
            w2: Linear::new(hidden_dim, dim),
        }
    }

    pub fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        let h = self.w1.forward(x);
        let act = TensorGraph::<B>::gelu(&h);
        self.w2.forward(&act)
    }

    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        let mut params = self.w1.parameters();
        params.extend(self.w2.parameters());
        params
    }
}

// THE TRANSFORMER BLOCK

/// A single Transformer Block comprising an Attention mechanism followed by a Feed-Forward network.
pub struct TransformerBlock<B: Backend> {
    pub norm1: RMSNorm<B>,
    pub mha: MultiHeadAttention<B>,
    pub norm2: RMSNorm<B>,
    pub ffn: FeedForward<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(dim: usize, heads: usize) -> Self {
        Self {
            norm1: RMSNorm::new(dim),
            mha: MultiHeadAttention::new(dim, heads),
            norm2: RMSNorm::new(dim),
            ffn: FeedForward::new(dim, dim * 4),
        }
    }

    pub fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        let n1 = self.norm1.forward(x);
        let attn = self.mha.forward(&n1, &n1, &n1);
        let res1 = TensorGraph::<B>::add(x, &attn);

        let n2 = self.norm2.forward(&res1);
        let ffn_out = self.ffn.forward(&n2);
        TensorGraph::<B>::add(&res1, &ffn_out)
    }

    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        let mut params = self.norm1.parameters();
        params.extend(self.norm2.parameters());
        params.extend(self.ffn.parameters());
        params
    }
}

// POOLING LAYERS
/// Max pooling operation for dimensionality reduction in CNNs.
pub struct MaxPool2d {
    pub kernel_size: usize,
}
impl MaxPool2d {
    pub fn new(kernel_size: usize) -> Self {
        Self { kernel_size }
    }
    pub fn forward<B: Backend>(&self, x: &TensorNode<B>) -> TensorNode<B> {
        B::max_pool2d(x, self.kernel_size)
    }
}

/// Average pooling operation for dimensionality reduction.
pub struct AvgPool2d {
    pub kernel_size: usize,
}
impl AvgPool2d {
    pub fn new(kernel_size: usize) -> Self {
        Self { kernel_size }
    }
    pub fn forward<B: Backend>(&self, x: &TensorNode<B>) -> TensorNode<B> {
        B::avg_pool2d(x, self.kernel_size)
    }
}

// BATCH NORMALIZATION

/// Batch Normalization for stabilizing neural network training.
pub struct BatchNorm2d<B: Backend> {
    pub gamma: TensorNode<B>,
    pub beta: TensorNode<B>,
    pub running_mean: TensorNode<B>,
    pub running_var: TensorNode<B>,
    pub momentum: f32,
}

impl<B: Backend> BatchNorm2d<B> {
    pub fn new(channels: usize) -> Self {
        Self {
            gamma: TensorGraph::<B>::new_cpu(vec![1.0; channels], vec![1, channels]),
            beta: TensorGraph::<B>::new_cpu(vec![0.0; channels], vec![1, channels]),
            running_mean: TensorGraph::<B>::new_cpu(vec![0.0; channels], vec![1, channels]),
            running_var: TensorGraph::<B>::new_cpu(vec![1.0; channels], vec![1, channels]),
            momentum: 0.1,
        }
    }

    pub fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        B::batch_norm(
            x,
            &self.gamma,
            &self.beta,
            &self.running_mean,
            &self.running_var,
            self.momentum,
        )
    }

    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

// ADVANCED CONVOLUTIONS

pub struct Conv1d<B: Backend> {
    pub weight: TensorNode<B>,
}

impl<B: Backend> Conv1d<B> {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            weight: TensorGraph::<B>::kaiming_random(out_channels, in_channels * kernel_size),
        }
    }
    pub fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        B::conv1d(x, &self.weight)
    }
    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        vec![self.weight.clone()]
    }
}

pub struct Conv3d<B: Backend> {
    pub weight: TensorNode<B>,
}

impl<B: Backend> Conv3d<B> {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            weight: TensorGraph::<B>::kaiming_random(
                out_channels,
                in_channels * kernel_size * kernel_size * kernel_size,
            ),
        }
    }
    pub fn forward(&self, x: &TensorNode<B>) -> TensorNode<B> {
        B::conv3d(x, &self.weight)
    }
    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        vec![self.weight.clone()]
    }
}

// RECURRENT NEURAL NETWORKS
// Built natively using generic Ops
pub struct LSTM<B: Backend> {
    pub w_ii: Linear<B>,
    pub w_hi: Linear<B>, // Input Gate
    pub w_if: Linear<B>,
    pub w_hf: Linear<B>, // Forget Gate
    pub w_ig: Linear<B>,
    pub w_hg: Linear<B>, // Cell Gate
    pub w_io: Linear<B>,
    pub w_ho: Linear<B>, // Output Gate
}

impl<B: Backend> LSTM<B> {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            w_ii: Linear::new(input_dim, hidden_dim),
            w_hi: Linear::new(hidden_dim, hidden_dim),
            w_if: Linear::new(input_dim, hidden_dim),
            w_hf: Linear::new(hidden_dim, hidden_dim),
            w_ig: Linear::new(input_dim, hidden_dim),
            w_hg: Linear::new(hidden_dim, hidden_dim),
            w_io: Linear::new(input_dim, hidden_dim),
            w_ho: Linear::new(hidden_dim, hidden_dim),
        }
    }

    /// Single Step Forward: Takes (Input, Previous Hidden, Previous Cell)
    pub fn forward_step(
        &self,
        x: &TensorNode<B>,
        h_prev: &TensorNode<B>,
        c_prev: &TensorNode<B>,
    ) -> (TensorNode<B>, TensorNode<B>) {
        // i_t = sigmoid(W_ii * x + b_ii + W_hi * h_prev + b_hi)
        let i_t = B::sigmoid(&TensorGraph::<B>::add(
            &self.w_ii.forward(x),
            &self.w_hi.forward(h_prev),
        ));

        // f_t = sigmoid(W_if * x + b_if + W_hf * h_prev + b_hf)
        let f_t = B::sigmoid(&TensorGraph::<B>::add(
            &self.w_if.forward(x),
            &self.w_hf.forward(h_prev),
        ));

        // g_t = tanh(W_ig * x + b_ig + W_hg * h_prev + b_hg)
        let g_t = B::tanh(&TensorGraph::<B>::add(
            &self.w_ig.forward(x),
            &self.w_hg.forward(h_prev),
        ));

        // o_t = sigmoid(W_io * x + b_io + W_ho * h_prev + b_ho)
        let o_t = B::sigmoid(&TensorGraph::<B>::add(
            &self.w_io.forward(x),
            &self.w_ho.forward(h_prev),
        ));

        // c_t = f_t * c_prev + i_t * g_t
        let c_t = TensorGraph::<B>::add(
            &TensorGraph::<B>::mul(&f_t, c_prev),
            &TensorGraph::<B>::mul(&i_t, &g_t),
        );

        // h_t = o_t * tanh(c_t)
        let h_t = TensorGraph::<B>::mul(&o_t, &B::tanh(&c_t));

        (h_t, c_t)
    }

    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        let mut p = Vec::new();
        p.extend(self.w_ii.parameters());
        p.extend(self.w_hi.parameters());
        p.extend(self.w_if.parameters());
        p.extend(self.w_hf.parameters());
        p.extend(self.w_ig.parameters());
        p.extend(self.w_hg.parameters());
        p.extend(self.w_io.parameters());
        p.extend(self.w_ho.parameters());
        p
    }
}

pub struct GRU<B: Backend> {
    pub w_ir: Linear<B>,
    pub w_hr: Linear<B>, // Reset Gate
    pub w_iz: Linear<B>,
    pub w_hz: Linear<B>, // Update Gate
    pub w_in: Linear<B>,
    pub w_hn: Linear<B>, // New Gate
}

impl<B: Backend> GRU<B> {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            w_ir: Linear::new(input_dim, hidden_dim),
            w_hr: Linear::new(hidden_dim, hidden_dim),
            w_iz: Linear::new(input_dim, hidden_dim),
            w_hz: Linear::new(hidden_dim, hidden_dim),
            w_in: Linear::new(input_dim, hidden_dim),
            w_hn: Linear::new(hidden_dim, hidden_dim),
        }
    }

    /// Single Step Forward: Takes (Input, Previous Hidden)
    pub fn forward_step(&self, x: &TensorNode<B>, h_prev: &TensorNode<B>) -> TensorNode<B> {
        // r_t = sigmoid(W_ir * x + b_ir + W_hr * h_prev + b_hr)
        let r_t = B::sigmoid(&TensorGraph::<B>::add(
            &self.w_ir.forward(x),
            &self.w_hr.forward(h_prev),
        ));

        // z_t = sigmoid(W_iz * x + b_iz + W_hz * h_prev + b_hz)
        let z_t = B::sigmoid(&TensorGraph::<B>::add(
            &self.w_iz.forward(x),
            &self.w_hz.forward(h_prev),
        ));

        // r_h = r_t * W_hn(h_prev) + b_hn
        let r_h = TensorGraph::<B>::mul(&r_t, &self.w_hn.forward(h_prev));

        // n_t = tanh(W_in * x + b_in + r_h)
        let n_t = B::tanh(&TensorGraph::<B>::add(&self.w_in.forward(x), &r_h));

        // h_t = (1 - z_t) * n_t + z_t * h_prev
        // Algebraically simplified as: h_t = n_t + z_t * (h_prev - n_t)
        let h_diff = TensorGraph::<B>::sub(h_prev, &n_t);
        let z_h_diff = TensorGraph::<B>::mul(&z_t, &h_diff);

        TensorGraph::<B>::add(&n_t, &z_h_diff)
    }

    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        let mut p = Vec::new();
        p.extend(self.w_ir.parameters());
        p.extend(self.w_hr.parameters());
        p.extend(self.w_iz.parameters());
        p.extend(self.w_hz.parameters());
        p.extend(self.w_in.parameters());
        p.extend(self.w_hn.parameters());
        p
    }
}

// HIGH-THROUGHPUT LLM SERVING (PagedAttention)
/// PagedAttention optimized for virtual memory caching.
pub struct PagedAttention<B: Backend> {
    pub w_q: Linear<B>,
    pub w_k: Linear<B>,
    pub w_v: Linear<B>,
    pub w_o: Linear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> PagedAttention<B> {
    pub fn new(dim: usize, num_heads: usize) -> Self {
        let head_dim = dim / num_heads;
        Self {
            w_q: Linear::new(dim, dim),
            w_k: Linear::new(dim, dim),
            w_v: Linear::new(dim, dim),
            w_o: Linear::new(dim, dim),
            num_heads,
            head_dim,
        }
    }

    /// Forward pass utilizing a virtual block table for fragmented KV-Cache memory
    pub fn forward(
        &self,
        x: &TensorNode<B>,
        kv_cache: &TensorNode<B>, // Physical memory pool: [num_blocks, block_size, num_heads, head_dim]
        block_tables: &TensorNode<B>, // Virtual memory mapping: [batch_size, max_blocks_per_seq]
        context_lens: &TensorNode<B>, // Length of each sequence in the batch
    ) -> TensorNode<B> {
        let q = self.w_q.forward(x);
        let k = self.w_k.forward(x);
        let v = self.w_v.forward(x);

        // Pushes the PagedAttention operation to the compute graph for WGSL fusion
        let attention_out = B::paged_attention(&q, &k, &v, kv_cache, block_tables, context_lens);

        self.w_o.forward(&attention_out)
    }

    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        let mut p = self.w_q.parameters();
        p.extend(self.w_k.parameters());
        p.extend(self.w_v.parameters());
        p.extend(self.w_o.parameters());
        p
    }
}
