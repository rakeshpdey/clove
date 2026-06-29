/*
 * src/nn/embedding.rs
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

//! Embedding layer implementation for the Clove Engine.
//!
//! This module provides the `Embedding` layer, which acts as the entry point
//! for NLP models. It transforms discrete integer token indices into
//! dense, learnable floating-point vectors.

use crate::backend::Backend;
use crate::tensor::{TensorGraph, TensorNode};
use ndarray::Array2;

/// A lookup table that stores embeddings of a fixed dictionary and size.
/// This module is used to transform discrete token indices into dense vectors
/// of a fixed size, which are then passed into the Transformer layers.
pub struct Embedding<B: Backend> {
    /// The learned embedding weights [vocab_size, hidden_dim].
    pub weights: TensorNode<B>,
    /// The size of the dictionary of embeddings.
    pub vocab_size: usize,
    /// The size of each embedding vector.
    pub hidden_dim: usize,
}

impl<B: Backend> Embedding<B> {
    /// Creates a new Embedding layer with Kaiming random initialization.
    pub fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        Self {
            weights: TensorGraph::<B>::kaiming_random(vocab_size, hidden_dim),
            vocab_size,
            hidden_dim,
        }
    }

    /// Performs the embedding lookup.
    /// # Arguments
    /// * `indices` - An `Array2` containing the token indices to look up.
    pub fn forward(&self, indices: &Array2<f32>) -> TensorNode<B> {
        TensorGraph::<B>::embedding(&self.weights, indices)
    }

    /// Returns the trainable parameters of this layer.
    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        vec![self.weights.clone()]
    }
}
