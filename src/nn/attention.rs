/*
 * src/nn/attention.rs
 * Multi-Head Attention mechanism.
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

//! Multi-Head Attention mechanism for Transformer architectures.
//!
//! This module provides the standard Scaled Dot-Product Attention mechanism.
//!
//! # JIT Optimization
//! Note that the mathematical operations defined in `forward` are not executed
//! as individual kernels. The Organon JIT Compiler's XLA Pattern Matcher
//! automatically detects the `MatMul(Softmax(MatMul))` sequence and fuses this
//! entire logic into a single, high-throughput FlashAttention WGSL kernel
//! at runtime.

use crate::backend::Backend;
use crate::tensor::{TensorGraph, TensorNode};
use std::marker::PhantomData;

// MULTI-HEAD ATTENTION MODULE
/// Implements the Scaled Dot-Product Attention mechanism.
/// The JIT Compiler automatically fuses this graph into FlashAttention.
pub struct MultiHeadAttention<B: Backend> {
    pub num_heads: usize,
    pub hidden_size: usize,
    pub head_dim: usize,
    _marker: PhantomData<B>,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Initializes the attention mechanism.
    /// head_dim is automatically inferred from hidden_size / num_heads.
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        Self {
            num_heads,
            hidden_size,
            head_dim: hidden_size / num_heads,
            _marker: PhantomData,
        }
    }

    /// Executes the forward pass: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    pub fn forward(
        &self,
        q: &TensorNode<B>,
        k: &TensorNode<B>,
        v: &TensorNode<B>,
    ) -> TensorNode<B> {
        // With No WGSL strings, no buffer mapping, no device queues.
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
