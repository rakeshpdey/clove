/*
 * src/ffi.rs
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

use crate::backend::{Backend, WgpuBackend};
use crate::optim::AdamW;
use crate::tensor::TensorNode;
use ndarray::Array2;
use std::ffi::c_void;
use std::os::raw::c_float;

/// Opaque pointer representing a `TensorNode<WgpuBackend>` in C.
pub type CTensor = *mut c_void;

/// Opaque pointer representing the `AdamW` optimizer in C.
pub type COptimizer = *mut c_void;

// TENSOR CREATION & MEMORY MANAGEMENT

/// Creates a new Tensor from raw CPU data.
///
/// # Safety
/// The `data` pointer must point to a contiguous block of memory containing
/// at least `rows * cols` elements of type `c_float`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_tensor_create(
    data: *const c_float,
    rows: usize,
    cols: usize,
) -> CTensor {
    let slice = unsafe { std::slice::from_raw_parts(data, rows * cols) };
    let array = Array2::from_shape_vec((rows, cols), slice.to_vec()).unwrap();
    let tensor = WgpuBackend::new(array);
    Box::into_raw(Box::new(tensor)) as CTensor
}

/// Safely frees the Tensor memory once the foreign language is done with it.
///
/// # Safety
/// The `tensor_ptr` must be a valid pointer originating from a tensor creation
/// or operation function within this library. Double-freeing results in undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_tensor_free(tensor_ptr: CTensor) {
    if !tensor_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(tensor_ptr as *mut TensorNode<WgpuBackend>);
        }
    }
}

/// Copies the tensor data back into a pre-allocated raw C-array.
///
/// # Safety
/// The `tensor_ptr` must be a valid tensor pointer. The `out_data` pointer must point
/// to a valid, writable block of memory large enough to hold all elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_tensor_data(tensor_ptr: CTensor, out_data: *mut c_float) {
    if tensor_ptr.is_null() || out_data.is_null() {
        return;
    }
    let tensor = unsafe { &*(tensor_ptr as *const TensorNode<WgpuBackend>) };
    let array = WgpuBackend::to_cpu(&tensor.read().unwrap());
    let (vec_data, _) = array.into_raw_vec_and_offset();
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_data, vec_data.len()) };
    out_slice.copy_from_slice(&vec_data);
}

// BASIC MATH & SHAPE OPERATIONS

/// Performs matrix multiplication on two tensors.
///
/// # Safety
/// Both `a_ptr` and `b_ptr` must be valid `CTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_matmul(a_ptr: CTensor, b_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    let b = unsafe { &*(b_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::matmul(a, b))) as CTensor
}

/// Adds two tensors.
///
/// # Safety
/// Both `a_ptr` and `b_ptr` must be valid `CTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_add(a_ptr: CTensor, b_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    let b = unsafe { &*(b_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::add(a, b))) as CTensor
}

/// Subtracts tensor b from a.
///
/// # Safety
/// Both `a_ptr` and `b_ptr` must be valid `CTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_sub(a_ptr: CTensor, b_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    let b = unsafe { &*(b_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::sub(a, b))) as CTensor
}

/// Multiplies two tensors element-wise.
///
/// # Safety
/// Both `a_ptr` and `b_ptr` must be valid `CTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_mul(a_ptr: CTensor, b_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    let b = unsafe { &*(b_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::mul(a, b))) as CTensor
}

/// Multiplies a tensor by a scalar value.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_mul_scalar(a_ptr: CTensor, scalar: c_float) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::mul_scalar(a, scalar))) as CTensor
}

/// Transposes a tensor.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_transpose(a_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::transpose(a))) as CTensor
}

/// Flattens a tensor.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_flatten(a_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::flatten(a))) as CTensor
}

/// Concatenates two tensors.
///
/// # Safety
/// Both `a_ptr` and `b_ptr` must be valid `CTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_concat(a_ptr: CTensor, b_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    let b = unsafe { &*(b_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::concat_seq(a, b))) as CTensor
}

// ACTIVATIONS & TRIGONOMETRY

/// Applies the ReLU activation function.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_relu(a_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::relu(a))) as CTensor
}

/// Applies the GELU activation function.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_gelu(a_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::gelu(a))) as CTensor
}

/// Applies the Sigmoid activation function.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_sigmoid(a_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::sigmoid(a))) as CTensor
}

/// Applies the Tanh activation function.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_tanh(a_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::tanh(a))) as CTensor
}

/// Applies the Softmax activation function.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_softmax(a_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::softmax(a))) as CTensor
}

/// Applies the Sine trigonometric function.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_sin(a_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::sin(a))) as CTensor
}

/// Applies the Cosine trigonometric function.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_cos(a_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::cos(a))) as CTensor
}

// NEURAL NETWORK LAYERS

/// Computes layer normalization.
///
/// # Safety
/// All pointers (`a_ptr`, `gamma_ptr`, `beta_ptr`) must be valid `CTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_layer_norm(
    a_ptr: CTensor,
    gamma_ptr: CTensor,
    beta_ptr: CTensor,
) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    let g = unsafe { &*(gamma_ptr as *const TensorNode<WgpuBackend>) };
    let b = unsafe { &*(beta_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::layer_norm(a, g, b))) as CTensor
}

/// Computes batch normalization.
///
/// # Safety
/// All tensor pointers must be valid `CTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_batch_norm(
    x_ptr: CTensor,
    g_ptr: CTensor,
    b_ptr: CTensor,
    rm_ptr: CTensor,
    rv_ptr: CTensor,
    momentum: c_float,
) -> CTensor {
    let x = unsafe { &*(x_ptr as *const TensorNode<WgpuBackend>) };
    let g = unsafe { &*(g_ptr as *const TensorNode<WgpuBackend>) };
    let b = unsafe { &*(b_ptr as *const TensorNode<WgpuBackend>) };
    let rm = unsafe { &*(rm_ptr as *const TensorNode<WgpuBackend>) };
    let rv = unsafe { &*(rv_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::batch_norm(x, g, b, rm, rv, momentum))) as CTensor
}

/// Applies dropout to a tensor.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_dropout(a_ptr: CTensor, rate: c_float) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::dropout(a, rate))) as CTensor
}

/// Computes a 1D convolution.
///
/// # Safety
/// Both `i_ptr` and `k_ptr` must be valid `CTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_conv1d(i_ptr: CTensor, k_ptr: CTensor) -> CTensor {
    let i = unsafe { &*(i_ptr as *const TensorNode<WgpuBackend>) };
    let k = unsafe { &*(k_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::conv1d(i, k))) as CTensor
}

/// Computes a 2D convolution.
///
/// # Safety
/// Both `i_ptr` and `k_ptr` must be valid `CTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_conv2d(i_ptr: CTensor, k_ptr: CTensor) -> CTensor {
    let i = unsafe { &*(i_ptr as *const TensorNode<WgpuBackend>) };
    let k = unsafe { &*(k_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::conv2d(i, k))) as CTensor
}

/// Computes a 3D convolution.
///
/// # Safety
/// Both `i_ptr` and `k_ptr` must be valid `CTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_conv3d(i_ptr: CTensor, k_ptr: CTensor) -> CTensor {
    let i = unsafe { &*(i_ptr as *const TensorNode<WgpuBackend>) };
    let k = unsafe { &*(k_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::conv3d(i, k))) as CTensor
}

/// Applies 2D Max Pooling.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_max_pool2d(a_ptr: CTensor, kernel: usize) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::max_pool2d(a, kernel))) as CTensor
}

/// Applies 2D Average Pooling.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_avg_pool2d(a_ptr: CTensor, kernel: usize) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::avg_pool2d(a, kernel))) as CTensor
}

// ATTENTION & ADVANCED

/// Applies Rotary Positional Embeddings (RoPE).
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_rope(a_ptr: CTensor, pos_offset: usize, head_dim: usize) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::rope(a, pos_offset, head_dim))) as CTensor
}

/// Performs an embedding lookup.
///
/// # Safety
/// `w_ptr` must be a valid `CTensor` pointer.
/// `indices_ptr` must point to a valid, contiguous array of `rows * cols` elements of type `c_float`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_embedding(
    w_ptr: CTensor,
    indices_ptr: *const c_float,
    rows: usize,
    cols: usize,
) -> CTensor {
    let w = unsafe { &*(w_ptr as *const TensorNode<WgpuBackend>) };
    let slice = unsafe { std::slice::from_raw_parts(indices_ptr, rows * cols) };
    let indices_array = Array2::from_shape_vec((rows, cols), slice.to_vec()).unwrap();
    Box::into_raw(Box::new(WgpuBackend::embedding(w, &indices_array))) as CTensor
}

/// Executes vLLM-style Paged Attention.
///
/// # Safety
/// All provided tensor pointers must be valid `CTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_paged_attention(
    q_ptr: CTensor,
    k_ptr: CTensor,
    v_ptr: CTensor,
    kv_ptr: CTensor,
    bt_ptr: CTensor,
    cl_ptr: CTensor,
) -> CTensor {
    let q = unsafe { &*(q_ptr as *const TensorNode<WgpuBackend>) };
    let k = unsafe { &*(k_ptr as *const TensorNode<WgpuBackend>) };
    let v = unsafe { &*(v_ptr as *const TensorNode<WgpuBackend>) };
    let kv = unsafe { &*(kv_ptr as *const TensorNode<WgpuBackend>) };
    let bt = unsafe { &*(bt_ptr as *const TensorNode<WgpuBackend>) };
    let cl = unsafe { &*(cl_ptr as *const TensorNode<WgpuBackend>) };
    Box::into_raw(Box::new(WgpuBackend::paged_attention(q, k, v, kv, bt, cl))) as CTensor
}

/// Executes a TopK selection for routing.
///
/// # Safety
/// `a_ptr` must be a valid `CTensor` pointer.
/// `out_vals` and `out_idxs` must be valid, writable pointers to receive the resulting `CTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_topk(
    a_ptr: CTensor,
    k: usize,
    out_vals: *mut CTensor,
    out_idxs: *mut CTensor,
) {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    let (vals, idxs) = WgpuBackend::topk(a, k);
    unsafe {
        *out_vals = Box::into_raw(Box::new(vals)) as CTensor;
        *out_idxs = Box::into_raw(Box::new(idxs)) as CTensor;
    }
}

// LOSS FUNCTIONS

/// Computes the Cross Entropy Loss.
///
/// # Safety
/// `l_ptr` must be a valid `CTensor` pointer.
/// `targets_ptr` must point to a valid, contiguous array of `rows * cols` elements of type `c_float`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_cross_entropy(
    l_ptr: CTensor,
    targets_ptr: *const c_float,
    rows: usize,
    cols: usize,
) -> CTensor {
    let l = unsafe { &*(l_ptr as *const TensorNode<WgpuBackend>) };
    let slice = unsafe { std::slice::from_raw_parts(targets_ptr, rows * cols) };
    let t_array = Array2::from_shape_vec((rows, cols), slice.to_vec()).unwrap();
    Box::into_raw(Box::new(WgpuBackend::cross_entropy(l, &t_array))) as CTensor
}

/// Computes the Mean Squared Error (MSE) Loss.
///
/// # Safety
/// `p_ptr` must be a valid `CTensor` pointer.
/// `targets_ptr` must point to a valid, contiguous array of `rows * cols` elements of type `c_float`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_mse(
    p_ptr: CTensor,
    targets_ptr: *const c_float,
    rows: usize,
    cols: usize,
) -> CTensor {
    let p = unsafe { &*(p_ptr as *const TensorNode<WgpuBackend>) };
    let slice = unsafe { std::slice::from_raw_parts(targets_ptr, rows * cols) };
    let t_array = Array2::from_shape_vec((rows, cols), slice.to_vec()).unwrap();
    Box::into_raw(Box::new(WgpuBackend::mse(p, &t_array))) as CTensor
}

/// Computes the Huber Loss.
///
/// # Safety
/// `p_ptr` must be a valid `CTensor` pointer.
/// `targets_ptr` must point to a valid, contiguous array of `rows * cols` elements of type `c_float`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_huber_loss(
    p_ptr: CTensor,
    targets_ptr: *const c_float,
    rows: usize,
    cols: usize,
    delta: c_float,
) -> CTensor {
    let p = unsafe { &*(p_ptr as *const TensorNode<WgpuBackend>) };
    let slice = unsafe { std::slice::from_raw_parts(targets_ptr, rows * cols) };
    let t_array = Array2::from_shape_vec((rows, cols), slice.to_vec()).unwrap();
    Box::into_raw(Box::new(WgpuBackend::huber_loss(p, &t_array, delta))) as CTensor
}

/// Computes the Binary Cross Entropy with Logits Loss.
///
/// # Safety
/// `p_ptr` must be a valid `CTensor` pointer.
/// `targets_ptr` must point to a valid, contiguous array of `rows * cols` elements of type `c_float`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_bce_with_logits(
    p_ptr: CTensor,
    targets_ptr: *const c_float,
    rows: usize,
    cols: usize,
) -> CTensor {
    let p = unsafe { &*(p_ptr as *const TensorNode<WgpuBackend>) };
    let slice = unsafe { std::slice::from_raw_parts(targets_ptr, rows * cols) };
    let t_array = Array2::from_shape_vec((rows, cols), slice.to_vec()).unwrap();
    Box::into_raw(Box::new(WgpuBackend::bce_with_logits(p, &t_array))) as CTensor
}

// AUTOGRAD & OPTIMIZER

/// Triggers the backward pass (autograd).
///
/// # Safety
/// `tensor_ptr` must be a valid `CTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_backward(tensor_ptr: CTensor) {
    if tensor_ptr.is_null() {
        return;
    }
    let tensor = unsafe { &*(tensor_ptr as *const TensorNode<WgpuBackend>) };
    crate::tensor::TensorGraph::backward(tensor);
}

/// Creates a GPU-Accelerated AdamW Optimizer.
///
/// # Safety
/// `params_array` must point to a valid, contiguous array of `num_params` elements of type `CTensor`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_optimizer_create(
    lr: c_float,
    params_array: *const CTensor,
    num_params: usize,
) -> COptimizer {
    let ptrs = unsafe { std::slice::from_raw_parts(params_array, num_params) };
    let mut params = Vec::new();
    for &ptr in ptrs {
        if !ptr.is_null() {
            let node = unsafe { &*(ptr as *const TensorNode<WgpuBackend>) };
            params.push(node.clone());
        }
    }

    let opt = AdamW::new(lr, params);
    Box::into_raw(Box::new(opt)) as COptimizer
}

/// Executes a single gradient descent step on the GPU.
///
/// # Safety
/// `opt_ptr` must be a valid `COptimizer` pointer created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_optimizer_step(opt_ptr: COptimizer) {
    if opt_ptr.is_null() {
        return;
    }
    let opt = unsafe { &mut *(opt_ptr as *mut AdamW) };
    opt.step();
}

/// Zeros the gradients of all parameters held by the optimizer.
///
/// # Safety
/// `opt_ptr` must be a valid `COptimizer` pointer created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_optimizer_zero_grad(opt_ptr: COptimizer) {
    if opt_ptr.is_null() {
        return;
    }
    let opt = unsafe { &mut *(opt_ptr as *mut AdamW) };
    opt.zero_grad();
}

/// Frees the Optimizer memory.
///
/// # Safety
/// `opt_ptr` must be a valid `COptimizer` pointer created by this library. Double-freeing results in undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_optimizer_free(opt_ptr: COptimizer) {
    if !opt_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(opt_ptr as *mut AdamW);
        }
    }
}
