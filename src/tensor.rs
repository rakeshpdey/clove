/*
 * src/tensor.rs
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

//! Core Tensor and Autograd Engine for the Clove Engine.
//!
//! This module defines the `TensorGraph`, the fundamental data structure
//! used for all computations. It implements the Reverse-Mode Automatic
//! Differentiation (Autograd) system, including topological graph traversal,
//! gradient accumulation, and memory-efficient recomputation (checkpointing).

use ndarray::Array2;
use std::cell::Cell;
use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use crate::backend::{Backend, WgpuBackend};
pub use crate::backend::{Precision, TensorData};

// GLOBAL AUTOGRAD STATE MANAGEMENT
// Thread-local storage for controlling gradient tracking.
// Uses Cell for zero-overhead, non-blocking state access in the forward pass.
thread_local! {
    pub static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Returns true if autograd is currently tracking operations.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|e| e.get()) // <-- FIX: Uses .get() which has zero overhead
}

/// A scope guard that temporarily disables gradient tracking.
/// Useful for inference or recomputation checkpoints.
pub struct NoGradGuard;
impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl NoGradGuard {
    /// Disables autograd until the guard is dropped.
    pub fn new() -> Self {
        GRAD_ENABLED.with(|e| e.set(false));
        Self
    }
}

impl Drop for NoGradGuard {
    /// Restores autograd state automatically when the guard goes out of scope.
    fn drop(&mut self) {
        GRAD_ENABLED.with(|e| e.set(true));
    }
}

// TENSOR TYPES
/// Alias for a shared, thread-safe Tensor Node in the compute graph.
pub type TensorNode<B> = Arc<RwLock<TensorGraph<B>>>;

/// A closure representing the backward gradient computation function.
pub type BackwardOp<B> = Box<dyn Fn(&TensorGraph<B>) + Send + Sync>;

/// Shorthand aliases for the primary Wgpu implementation.
pub type Node = TensorNode<WgpuBackend>;
pub type Tensor = TensorGraph<WgpuBackend>;

/// The core Tensor structure within the computation graph.
pub struct TensorGraph<B: Backend> {
    pub data: B::TensorPrimitive,
    pub shape: Vec<usize>,
    pub grad: Option<B::TensorPrimitive>,
    pub backward: Option<BackwardOp<B>>,
    pub creators: Vec<TensorNode<B>>,
    pub device: B::Device,
}

impl<B: Backend> TensorGraph<B> {
    /// Creates a tensor node from raw CPU vector data.
    pub fn new_cpu(data: Vec<f32>, shape: Vec<usize>) -> TensorNode<B> {
        B::new_cpu(data, shape)
    }

    /// Creates a tensor node from an ndarray.
    pub fn new(data: Array2<f32>) -> TensorNode<B> {
        B::new(data)
    }

    /// Initializes a tensor with Kaiming random values.
    pub fn kaiming_random(in_feat: usize, out_feat: usize) -> TensorNode<B> {
        B::kaiming_random(in_feat, out_feat)
    }

    pub fn get_cpu_grad(&self) -> &[f32] {
        B::get_cpu_grad(self)
    }
    pub fn add_cpu_grad(&mut self, new_grad: &[f32]) {
        B::add_cpu_grad(self, new_grad)
    }
    pub fn clip_gradients(&mut self) {
        B::clip_gradients(self)
    }
    pub fn to_cpu(&self) -> Array2<f32> {
        B::to_cpu(self)
    }
    pub fn to_gpu(&mut self, device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) {
        B::to_gpu(self, device, queue)
    }
    pub fn grad_to_cpu(&self) -> Option<Array2<f32>> {
        B::grad_to_cpu(self)
    }

    // Math Opcodes
    pub fn matmul(a: &TensorNode<B>, b: &TensorNode<B>) -> TensorNode<B> {
        B::matmul(a, b)
    }
    pub fn add(a: &TensorNode<B>, b: &TensorNode<B>) -> TensorNode<B> {
        B::add(a, b)
    }
    pub fn sub(a: &TensorNode<B>, b: &TensorNode<B>) -> TensorNode<B> {
        B::sub(a, b)
    }
    pub fn mul(a: &TensorNode<B>, b: &TensorNode<B>) -> TensorNode<B> {
        B::mul(a, b)
    }
    pub fn mul_scalar(a: &TensorNode<B>, scalar: f32) -> TensorNode<B> {
        B::mul_scalar(a, scalar)
    }
    pub fn transpose(a: &TensorNode<B>) -> TensorNode<B> {
        B::transpose(a)
    }
    pub fn flatten(a: &TensorNode<B>) -> TensorNode<B> {
        B::flatten(a)
    }
    pub fn concat_seq(a: &TensorNode<B>, b: &TensorNode<B>) -> TensorNode<B> {
        B::concat_seq(a, b)
    }

    // Activation Opcodes
    pub fn relu(a: &TensorNode<B>) -> TensorNode<B> {
        B::relu(a)
    }
    pub fn gelu(a: &TensorNode<B>) -> TensorNode<B> {
        B::gelu(a)
    }
    pub fn sin(a: &TensorNode<B>) -> TensorNode<B> {
        B::sin(a)
    }
    pub fn cos(a: &TensorNode<B>) -> TensorNode<B> {
        B::cos(a)
    }
    pub fn softmax(a: &TensorNode<B>) -> TensorNode<B> {
        B::softmax(a)
    }
    pub fn layer_norm(
        a: &TensorNode<B>,
        gamma: &TensorNode<B>,
        beta: &TensorNode<B>,
    ) -> TensorNode<B> {
        B::layer_norm(a, gamma, beta)
    }
    pub fn dropout(a: &TensorNode<B>, rate: f32) -> TensorNode<B> {
        B::dropout(a, rate)
    }
    pub fn conv2d(i: &TensorNode<B>, k: &TensorNode<B>) -> TensorNode<B> {
        B::conv2d(i, k)
    }
    pub fn embedding(w: &TensorNode<B>, indices: &Array2<f32>) -> TensorNode<B> {
        B::embedding(w, indices)
    }
    pub fn cross_entropy(l: &TensorNode<B>, targets: &Array2<f32>) -> TensorNode<B> {
        B::cross_entropy(l, targets)
    }
    pub fn mse(p: &TensorNode<B>, targets: &Array2<f32>) -> TensorNode<B> {
        B::mse(p, targets)
    }
    pub fn rope(a: &TensorNode<B>, pos_offset: usize, head_dim: usize) -> TensorNode<B> {
        B::rope(a, pos_offset, head_dim)
    }

    // API Endpoint for Automatic Mixed Precision
    pub fn cast(a: &TensorNode<B>, precision: Precision) -> TensorNode<B> {
        B::cast(a, precision)
    }
    pub fn topk(a: &TensorNode<B>, k: usize) -> (TensorNode<B>, TensorNode<B>) {
        B::topk(a, k)
    }

    // GRADIENT CHECKPOINTING API

    /// Executes a function while disabling gradient tracking for memory efficiency,
    /// then registers a recomputation hook for the backward pass.
    pub fn checkpoint<F>(func: F, input: &TensorNode<B>) -> TensorNode<B>
    where
        F: Fn(&TensorNode<B>) -> TensorNode<B> + Send + Sync + 'static,
    {
        // Forward pass with no memory tracking
        let output = {
            let _guard = NoGradGuard::new();
            func(input)
        };

        if !is_grad_enabled() {
            return output;
        }

        // Stitching the graph
        let mut out_write = output.write().unwrap();
        out_write.creators = vec![Arc::clone(input)];

        let func_arc = Arc::new(func);
        let input_clone = Arc::clone(input);

        // Custom Recomputation Backward Hook
        out_write.backward = Some(Box::new(move |out_graph: &TensorGraph<B>| {
            let recomputed_out = func_arc(&input_clone);

            if let Some(incoming_grad) = &out_graph.grad {
                // Extract the device from the tensor graph node first
                let device = recomputed_out.read().unwrap().device.clone();

                // Pass both the incoming gradient primitive and the device reference
                recomputed_out.write().unwrap().grad =
                    Some(B::clone_tensor(incoming_grad, &device));
            }

            Self::backward_subgraph(&recomputed_out, &input_clone);
        }));

        drop(out_write);
        output
    }

    /// Performs a topological sort of the graph subset for recomputation.
    fn build_topo_subgraph(
        v: &TensorNode<B>,
        stop_ptr: usize,
        topo: &mut Vec<TensorNode<B>>,
        visited: &mut HashSet<usize>,
    ) {
        let ptr = Arc::as_ptr(v) as usize;
        if visited.contains(&ptr) {
            return;
        }

        visited.insert(ptr);

        if ptr != stop_ptr {
            for child in &v.read().unwrap().creators {
                Self::build_topo_subgraph(child, stop_ptr, topo, visited);
            }
        }
        topo.push(Arc::clone(v));
    }

    /// Backpropagates through a temporary subgraph constructed during checkpointing.
    fn backward_subgraph(node: &TensorNode<B>, stop_node: &TensorNode<B>) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        let stop_ptr = Arc::as_ptr(stop_node) as usize;

        Self::build_topo_subgraph(node, stop_ptr, &mut topo, &mut visited);

        for v in topo.into_iter().rev() {
            if Arc::as_ptr(&v) as usize == stop_ptr {
                continue;
            }

            let backward_closure = {
                let v_read = v.read().unwrap();
                v_read.backward.as_ref().map(|b| {
                    let ptr: *const (dyn Fn(&TensorGraph<B>) + Send + Sync) = b.as_ref();
                    ptr
                })
            };
            if let Some(bwd_ptr) = backward_closure {
                let v_read = v.read().unwrap();
                unsafe {
                    (*bwd_ptr)(&v_read);
                }
            }
        }
    }

    // BACKPROPAGATION ENGINE

    /// Helper to build a full topological ordering of the compute graph.
    fn build_topo(v: &TensorNode<B>, topo: &mut Vec<TensorNode<B>>, visited: &mut HashSet<usize>) {
        let ptr = Arc::as_ptr(v) as usize;
        if !visited.contains(&ptr) {
            visited.insert(ptr);
            for child in &v.read().unwrap().creators {
                Self::build_topo(child, topo, visited);
            }
            topo.push(Arc::clone(v));
        }
    }

    /// Performs the full backward pass starting from the root node.
    pub fn backward(node: &TensorNode<B>) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        Self::build_topo(node, &mut topo, &mut visited);

        {
            let mut root = node.write().unwrap();
            let total_elements = root.shape.iter().product::<usize>();
            root.grad = Some(B::ones(total_elements, &root.device));
        }

        for v in topo.into_iter().rev() {
            let backward_closure = {
                let v_read = v.read().unwrap();
                v_read.backward.as_ref().map(|b| {
                    let ptr: *const (dyn Fn(&TensorGraph<B>) + Send + Sync) = b.as_ref();
                    ptr
                })
            };
            if let Some(bwd_ptr) = backward_closure {
                let v_read = v.read().unwrap();
                unsafe {
                    (*bwd_ptr)(&v_read);
                }
            }
        }
    }
}
