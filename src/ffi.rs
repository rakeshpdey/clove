/*
 * src/ffi.rs
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Rakesh Dey
 *
 * Licensed under the MIT License <LICENSE-MIT or http://opensource.org/licenses/MIT>.
 * 
 * Note: Portions of this software are adapted from existing open-source frameworks.
 * This file may not be copied, modified, or distributed except according to the terms
 * of the MIT license.
 */

use crate::backend::{Backend, WgpuBackend};
use crate::tensor::TensorNode;
use ndarray::Array2;
use std::ffi::c_void;
use std::os::raw::c_float;

/// Opaque pointer representing a `TensorNode<WgpuBackend>` in C.
/// This hides the Rust-specific internal structure from the foreign caller.
pub type CTensor = *mut c_void;

// ========================================================================
// TENSOR CREATION & MEMORY MANAGEMENT
// ========================================================================

/// Creates a new Tensor from raw CPU data (e.g., from a Python NumPy array).
/// 
/// # Safety
/// The `data` pointer must point to a contiguous block of memory containing 
/// at least `rows * cols` elements of type `c_float`. The caller is responsible
/// for ensuring the memory is not freed while this tensor is in use.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_tensor_create(data: *const c_float, rows: usize, cols: usize) -> CTensor {
    let slice = unsafe { std::slice::from_raw_parts(data, rows * cols) };
    let array = Array2::from_shape_vec((rows, cols), slice.to_vec()).unwrap();
    
    // Create the tensor node and leak it to the heap to bypass Rust's ownership rules
    let tensor = WgpuBackend::new(array);
    Box::into_raw(Box::new(tensor)) as CTensor
}

/// Safely frees the Tensor memory once the foreign language is done with it.
/// 
/// # Safety
/// The `tensor_ptr` must be a valid pointer originating from a tensor creation 
/// or operation function within this library. Double-freeing the same pointer
/// will result in undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_tensor_free(tensor_ptr: CTensor) {
    if !tensor_ptr.is_null() {
        unsafe {
            // Re-take ownership of the pointer and let it drop natively
            let _ = Box::from_raw(tensor_ptr as *mut TensorNode<WgpuBackend>);
        }
    }
}

/// Copies the tensor data back into a pre-allocated raw C-array (e.g., to read back into Python).
/// 
/// # Safety
/// The `tensor_ptr` must be a valid tensor pointer.
/// The `out_data` pointer must point to a valid, writable block of memory 
/// large enough to hold all elements of the tensor.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_tensor_data(tensor_ptr: CTensor, out_data: *mut c_float) {
    if tensor_ptr.is_null() || out_data.is_null() { return; }
    
    let tensor = unsafe { &*(tensor_ptr as *const TensorNode<WgpuBackend>) };
    let array = WgpuBackend::to_cpu(&tensor.read().unwrap());
    
    let (vec_data, _) = array.into_raw_vec_and_offset();
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_data, vec_data.len()) };
    out_slice.copy_from_slice(&vec_data);
}

// ========================================================================
// CORE OPERATIONS (Exposed to Python/C++)
// ========================================================================

/// Performs matrix multiplication on two tensors.
/// 
/// # Safety
/// Both `a_ptr` and `b_ptr` must be valid `CTensor` pointers created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_matmul(a_ptr: CTensor, b_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    let b = unsafe { &*(b_ptr as *const TensorNode<WgpuBackend>) };
    
    let result = WgpuBackend::matmul(a, b);
    
    Box::into_raw(Box::new(result)) as CTensor
}

/// Adds two tensors together.
/// 
/// # Safety
/// Both `a_ptr` and `b_ptr` must be valid `CTensor` pointers created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_add(a_ptr: CTensor, b_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    let b = unsafe { &*(b_ptr as *const TensorNode<WgpuBackend>) };
    
    let result = WgpuBackend::add(a, b);
    
    Box::into_raw(Box::new(result)) as CTensor
}

/// Applies the ReLU activation function to a tensor.
/// 
/// # Safety
/// The `a_ptr` must be a valid `CTensor` pointer created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_relu(a_ptr: CTensor) -> CTensor {
    let a = unsafe { &*(a_ptr as *const TensorNode<WgpuBackend>) };
    let result = WgpuBackend::relu(a);
    Box::into_raw(Box::new(result)) as CTensor
}

// ========================================================================
// AUTOGRAD (Exposed to Python/C++)
// ========================================================================

/// Triggers the backward pass (autograd) starting from the given tensor.
/// 
/// # Safety
/// The `tensor_ptr` must be a valid `CTensor` pointer created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_backward(tensor_ptr: CTensor) {
    if tensor_ptr.is_null() { return; }
    let tensor = unsafe { &*(tensor_ptr as *const TensorNode<WgpuBackend>) };
    
    // Trigger the backward pass using the existing TensorGraph implementation
    crate::tensor::TensorGraph::backward(tensor);
}