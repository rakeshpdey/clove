/*
 * src/ffi.rs
 * C-ABI BRIDGE
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

// This directive prevents Clippy from complaining about missing `# Safety` blocks
// on every single exported C function, keeping the bridge clean and readable.
#![allow(clippy::missing_safety_doc)]

use crate::backend::{Backend, Precision, WgpuBackend};
use crate::data::DataLoader;
use crate::nn::LanguageModel;
use crate::optim::{AdamW, GradScaler};
use crate::tensor::{Node, TensorGraph};

use ndarray::Array2;
use std::ffi::{CStr, c_void};
use std::os::raw::{c_char, c_float, c_int};

// ==============================================================================
// OPAQUE POINTER TYPES
// ==============================================================================
// These type aliases create strongly-typed opaque pointers for the C-ABI.
// Foreign languages will hold these pointers without knowing their Rust internals.

pub type CTensor = *mut c_void;
pub type COptimizer = *mut c_void;
pub type CGradScaler = *mut c_void;
pub type CModel = *mut c_void;
pub type CDataLoader = *mut c_void;

// ==============================================================================
// INTERNAL FFI HELPERS (Rust 2024 Compliant)
// ==============================================================================

/// Safely casts a raw C pointer back into a Rust TensorNode reference.
#[inline(always)]
unsafe fn as_node(ptr: CTensor) -> &'static Node {
    unsafe { &*(ptr as *const Node) }
}

/// Consumes a Rust object, moves it to the heap, and leaks it to C as a raw pointer.
#[inline(always)]
fn into_raw<T>(obj: T) -> *mut c_void {
    Box::into_raw(Box::new(obj)) as *mut c_void
}

/// Converts a raw C float array into an `ndarray::Array2` for matrix operations.
unsafe fn ptr_to_array2(ptr: *const c_float, rows: usize, cols: usize) -> Array2<f32> {
    unsafe {
        let size = rows * cols;
        let slice = std::slice::from_raw_parts(ptr, size);
        Array2::from_shape_vec((rows, cols), slice.to_vec())
            .expect("Failed to construct Array2 from FFI pointer")
    }
}

// ==============================================================================
// TENSOR LIFECYCLE & MEMORY
// ==============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_tensor_create(
    data_ptr: *const c_float,
    rows: usize,
    cols: usize,
) -> CTensor {
    unsafe {
        if data_ptr.is_null() {
            return std::ptr::null_mut();
        }
        let size = rows * cols;
        let data = std::slice::from_raw_parts(data_ptr, size).to_vec();

        let tensor_node = WgpuBackend::new_cpu(data, vec![rows, cols]);
        into_raw(tensor_node)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_tensor_free(tensor_ptr: CTensor) {
    unsafe {
        if !tensor_ptr.is_null() {
            // Re-take ownership to drop and free memory
            let _ = Box::from_raw(tensor_ptr as *mut Node);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_tensor_data(tensor_ptr: CTensor, out_data: *mut c_float) {
    unsafe {
        if tensor_ptr.is_null() || out_data.is_null() {
            return;
        }
        let node = as_node(tensor_ptr);

        // Download memory from Backend to CPU
        let array = WgpuBackend::to_cpu(&node.read().unwrap());
        if let Some(slice) = array.as_slice() {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), out_data, slice.len());
        }
    }
}

// ==============================================================================
// CORE MATH & SHAPE LOGIC
// ==============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_add(a: CTensor, b: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::add(as_node(a), as_node(b))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_sub(a: CTensor, b: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::sub(as_node(a), as_node(b))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_mul(a: CTensor, b: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::mul(as_node(a), as_node(b))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_matmul(a: CTensor, b: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::matmul(as_node(a), as_node(b))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_mul_scalar(a: CTensor, scalar: c_float) -> CTensor {
    unsafe { into_raw(WgpuBackend::mul_scalar(as_node(a), scalar)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_transpose(a: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::transpose(as_node(a))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_flatten(a: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::flatten(as_node(a))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_concat(a: CTensor, b: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::concat_seq(as_node(a), as_node(b))) }
}

// ==============================================================================
// ACTIVATIONS & TRIGONOMETRY
// ==============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_relu(a: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::relu(as_node(a))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_gelu(a: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::gelu(as_node(a))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_sigmoid(a: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::sigmoid(as_node(a))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_tanh(a: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::tanh(as_node(a))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_softmax(a: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::softmax(as_node(a))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_sin(a: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::sin(as_node(a))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_cos(a: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::cos(as_node(a))) }
}

// ==============================================================================
// VISION (CNNs) & POOLING
// ==============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_conv1d(i: CTensor, k: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::conv1d(as_node(i), as_node(k))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_conv2d(i: CTensor, k: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::conv2d(as_node(i), as_node(k))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_conv3d(i: CTensor, k: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::conv3d(as_node(i), as_node(k))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_max_pool2d(a: CTensor, kernel: usize) -> CTensor {
    unsafe { into_raw(WgpuBackend::max_pool2d(as_node(a), kernel)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_avg_pool2d(a: CTensor, kernel: usize) -> CTensor {
    unsafe { into_raw(WgpuBackend::avg_pool2d(as_node(a), kernel)) }
}

// ==============================================================================
// LLMs, ATTENTION, & NLP
// ==============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_layer_norm(a: CTensor, gamma: CTensor, beta: CTensor) -> CTensor {
    unsafe {
        into_raw(WgpuBackend::layer_norm(
            as_node(a),
            as_node(gamma),
            as_node(beta),
        ))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_rope(a: CTensor, pos_offset: usize, head_dim: usize) -> CTensor {
    unsafe { into_raw(WgpuBackend::rope(as_node(a), pos_offset, head_dim)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_paged_attention(
    q: CTensor,
    k: CTensor,
    v: CTensor,
    kv: CTensor,
    bt: CTensor,
    cl: CTensor,
) -> CTensor {
    unsafe {
        into_raw(WgpuBackend::paged_attention(
            as_node(q),
            as_node(k),
            as_node(v),
            as_node(kv),
            as_node(bt),
            as_node(cl),
        ))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_embedding(
    w: CTensor,
    indices_ptr: *const c_float,
    rows: usize,
    cols: usize,
) -> CTensor {
    unsafe {
        let indices = ptr_to_array2(indices_ptr, rows, cols);
        into_raw(WgpuBackend::embedding(as_node(w), &indices))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_topk(
    a: CTensor,
    k: usize,
    out_vals: *mut CTensor,
    out_idxs: *mut CTensor,
) {
    unsafe {
        let (values, indices) = WgpuBackend::topk(as_node(a), k);
        *out_vals = into_raw(values);
        *out_idxs = into_raw(indices);
    }
}

// ==============================================================================
// REGULARIZATION & TRAINING OPS
// ==============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_dropout(a: CTensor, rate: c_float) -> CTensor {
    unsafe { into_raw(WgpuBackend::dropout(as_node(a), rate)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_cast(a: CTensor, precision: c_int) -> CTensor {
    unsafe {
        let p = match precision {
            16 => Precision::F16,
            160 => Precision::BF16,
            _ => Precision::F32,
        };
        into_raw(WgpuBackend::cast(as_node(a), p))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_batch_norm(
    x: CTensor,
    g: CTensor,
    b: CTensor,
    rm: CTensor,
    rv: CTensor,
    momentum: c_float,
) -> CTensor {
    unsafe {
        into_raw(WgpuBackend::batch_norm(
            as_node(x),
            as_node(g),
            as_node(b),
            as_node(rm),
            as_node(rv),
            momentum,
        ))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_all_reduce(a: CTensor) -> CTensor {
    unsafe { into_raw(WgpuBackend::all_reduce(as_node(a))) }
}

// ==============================================================================
// LOSS FUNCTIONS & AUTOGRAD
// ==============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_mse(
    p: CTensor,
    targets_ptr: *const c_float,
    rows: usize,
    cols: usize,
) -> CTensor {
    unsafe {
        let targets = ptr_to_array2(targets_ptr, rows, cols);
        into_raw(WgpuBackend::mse(as_node(p), &targets))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_cross_entropy(
    l: CTensor,
    targets_ptr: *const c_float,
    rows: usize,
    cols: usize,
) -> CTensor {
    unsafe {
        let targets = ptr_to_array2(targets_ptr, rows, cols);
        into_raw(WgpuBackend::cross_entropy(as_node(l), &targets))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_huber_loss(
    p: CTensor,
    targets_ptr: *const c_float,
    rows: usize,
    cols: usize,
    delta: c_float,
) -> CTensor {
    unsafe {
        let targets = ptr_to_array2(targets_ptr, rows, cols);
        into_raw(WgpuBackend::huber_loss(as_node(p), &targets, delta))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_bce_with_logits(
    p: CTensor,
    targets_ptr: *const c_float,
    rows: usize,
    cols: usize,
) -> CTensor {
    unsafe {
        let targets = ptr_to_array2(targets_ptr, rows, cols);
        into_raw(WgpuBackend::bce_with_logits(as_node(p), &targets))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_backward(tensor_ptr: CTensor) {
    unsafe {
        if !tensor_ptr.is_null() {
            TensorGraph::<WgpuBackend>::backward(as_node(tensor_ptr));
        }
    }
}

// ==============================================================================
// OPTIMIZERS & AMP
// ==============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_optimizer_create(
    lr: c_float,
    params_array: *const CTensor,
    num_params: usize,
) -> COptimizer {
    unsafe {
        let ptrs = std::slice::from_raw_parts(params_array, num_params);
        let mut params = Vec::new();
        for &ptr in ptrs {
            if !ptr.is_null() {
                params.push(as_node(ptr).clone());
            }
        }
        into_raw(AdamW::new(lr, params)) as COptimizer
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_optimizer_step(opt_ptr: COptimizer) {
    unsafe {
        if !opt_ptr.is_null() {
            let opt = &mut *(opt_ptr as *mut AdamW);
            opt.step();
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_optimizer_zero_grad(opt_ptr: COptimizer) {
    unsafe {
        if !opt_ptr.is_null() {
            let opt = &mut *(opt_ptr as *mut AdamW);
            opt.zero_grad();
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_optimizer_free(opt_ptr: COptimizer) {
    unsafe {
        if !opt_ptr.is_null() {
            let _ = Box::from_raw(opt_ptr as *mut AdamW);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_grad_scaler_create() -> CGradScaler {
    // GradScaler creation is safe, no unsafe block needed
    into_raw(GradScaler::new()) as CGradScaler
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_grad_scaler_scale(
    scaler_ptr: CGradScaler,
    loss: c_float,
) -> c_float {
    unsafe {
        if scaler_ptr.is_null() {
            return loss;
        }
        let scaler = &*(scaler_ptr as *const GradScaler);
        scaler.scale(loss)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_grad_scaler_step(
    scaler_ptr: CGradScaler,
    opt_ptr: COptimizer,
    has_nans: bool,
) {
    unsafe {
        if !scaler_ptr.is_null() && !opt_ptr.is_null() {
            let scaler = &mut *(scaler_ptr as *mut GradScaler);
            let opt = &mut *(opt_ptr as *mut AdamW);
            scaler.step(opt, has_nans);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_grad_scaler_free(scaler_ptr: CGradScaler) {
    unsafe {
        if !scaler_ptr.is_null() {
            let _ = Box::from_raw(scaler_ptr as *mut GradScaler);
        }
    }
}

// ==============================================================================
// HIGH-LEVEL MODULES (LANGUAGE MODEL)
// ==============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_nn_language_model_create(
    vocab_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
) -> CModel {
    let model = LanguageModel::<WgpuBackend>::new(vocab_size, hidden_dim, num_layers, num_heads);
    into_raw(model) as CModel
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_nn_language_model_forward(
    model_ptr: CModel,
    indices_ptr: *const c_float,
    rows: usize,
    cols: usize,
) -> CTensor {
    unsafe {
        if model_ptr.is_null() {
            return std::ptr::null_mut();
        }
        let model = &*(model_ptr as *const LanguageModel<WgpuBackend>);
        let indices = ptr_to_array2(indices_ptr, rows, cols);
        into_raw(model.forward(&indices)) as CTensor
    }
}

/// Fetches the model parameters. If `out_ptrs` is null, it simply returns the total count
/// so the caller can allocate a pointer array. If `out_ptrs` is valid, it populates it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_nn_language_model_parameters(
    model_ptr: CModel,
    out_ptrs: *mut CTensor,
) -> usize {
    unsafe {
        if model_ptr.is_null() {
            return 0;
        }
        let model = &*(model_ptr as *const LanguageModel<WgpuBackend>);
        let params = model.parameters();

        // Save the length BEFORE we consume the vector in the loop!
        let count = params.len();

        if !out_ptrs.is_null() {
            let out_slice = std::slice::from_raw_parts_mut(out_ptrs, count);
            for (i, p) in params.into_iter().enumerate() {
                out_slice[i] = into_raw(p) as CTensor;
            }
        }

        count
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_nn_language_model_save(
    model_ptr: CModel,
    path: *const c_char,
) -> bool {
    unsafe {
        if model_ptr.is_null() || path.is_null() {
            return false;
        }
        let model = &*(model_ptr as *const LanguageModel<WgpuBackend>);
        if let Ok(str_path) = CStr::from_ptr(path).to_str() {
            model.save_safetensors(str_path).is_ok()
        } else {
            false
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_nn_language_model_load(
    model_ptr: CModel,
    path: *const c_char,
) -> bool {
    unsafe {
        if model_ptr.is_null() || path.is_null() {
            return false;
        }
        let model = &*(model_ptr as *const LanguageModel<WgpuBackend>);
        if let Ok(str_path) = CStr::from_ptr(path).to_str() {
            model.load_safetensors(str_path).is_ok()
        } else {
            false
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_nn_language_model_free(model_ptr: CModel) {
    unsafe {
        if !model_ptr.is_null() {
            let _ = Box::from_raw(model_ptr as *mut LanguageModel<WgpuBackend>);
        }
    }
}

// ==============================================================================
// ASYNCHRONOUS DATA LOADER
// ==============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_dataloader_create(
    path: *const c_char,
    seq_len: usize,
    batch_size: usize,
) -> CDataLoader {
    unsafe {
        if path.is_null() {
            return std::ptr::null_mut();
        }
        if let Ok(str_path) = CStr::from_ptr(path).to_str() {
            // Bypass DataPipeline setup here for simplicity across C-ABI
            let loader = DataLoader::from_file(str_path, seq_len, batch_size, None);
            into_raw(loader) as CDataLoader
        } else {
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_dataloader_next(
    loader_ptr: CDataLoader,
    out_x: *mut CTensor,
    out_y: *mut CTensor,
) {
    unsafe {
        if loader_ptr.is_null() || out_x.is_null() || out_y.is_null() {
            return;
        }
        let loader = &mut *(loader_ptr as *mut DataLoader);
        let (x, y) = loader.next_batch();
        *out_x = into_raw(WgpuBackend::new(x)) as CTensor;
        *out_y = into_raw(WgpuBackend::new(y)) as CTensor;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn clove_dataloader_free(loader_ptr: CDataLoader) {
    unsafe {
        if !loader_ptr.is_null() {
            let _ = Box::from_raw(loader_ptr as *mut DataLoader);
        }
    }
}
