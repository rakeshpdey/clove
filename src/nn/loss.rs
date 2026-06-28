/*
 * src/nn/loss.rs
 *Loss function
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

//! Objective functions for the Organon Engine.
//!
//! This module implements standard loss functions required for supervised learning,
//! including classification objectives like `CrossEntropyLoss` and regression
//! objectives like `MSELoss`. These functions compute the error between the
//! model's predictions and the ground-truth targets, serving as the starting
//! point for the backward pass (autograd).

use crate::backend::Backend;
use crate::tensor::{TensorGraph, TensorNode};
use ndarray::Array2;

/// Implementation of the Cross-Entropy loss function for classification tasks.
/// This objective function measures the performance of a classification model
/// whose output is a probability value between 0 and 1.
pub struct CrossEntropyLoss;

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossEntropyLoss {
    /// Creates a new instance of the CrossEntropyLoss.
    pub fn new() -> Self {
        Self
    }

    /// Calculates the loss between predicted logits and target indices.
    ///
    /// # Arguments
    /// * `logits` - The output tensor from the final layer of the network.
    /// * `targets` - The ground-truth class labels as an `Array2`.
    pub fn forward<B: Backend>(
        &self,
        logits: &TensorNode<B>,
        targets: &Array2<f32>,
    ) -> TensorNode<B> {
        TensorGraph::<B>::cross_entropy(logits, targets)
    }
}

/// Implementation of the Mean Squared Error (MSE) loss function for regression tasks.
///
/// This objective function measures the average of the squares of the errors—that is,
/// the average squared difference between the estimated values and the actual value.
pub struct MSELoss;

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl MSELoss {
    /// Creates a new instance of the MSELoss.
    pub fn new() -> Self {
        Self
    }

    /// Calculates the MSE loss between predictions and target values.
    ///
    /// # Arguments
    /// * `preds` - The output tensor from the regression model.
    /// * `targets` - The ground-truth values as an `Array2`.
    pub fn forward<B: Backend>(
        &self,
        preds: &TensorNode<B>,
        targets: &Array2<f32>,
    ) -> TensorNode<B> {
        TensorGraph::<B>::mse(preds, targets)
    }
}
