/*
 * src/device.rs
 * Hardware Abstraction Layer
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

//! Hardware abstraction and device lifecycle management.
//!
//! This module provides a unified interface (`EngineDevice`) to manage
//! diverse compute backends, including local CPU threads, physical GPUs,
//! distributed multi-GPU clusters, and browser-based WebGPU environments.
//! It also hosts the `BufferPool` for efficient VRAM memory lifecycle management.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::Buffer;

// Conditionally bring in WASM requirements when compiling for the browser!
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Defines the compute hardware backend for the Organon Engine.
/// This abstraction allows the same graph to run on local CPUs,
/// high-performance GPUs, or edge-native WebAssembly environments.
#[derive(Clone)]
pub enum EngineDevice {
    /// Standard multi-core CPU execution using Rayon.
    Cpu { cores: usize },
    /// Dedicated GPU execution (Vulkan/Metal/DX12).
    Gpu {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    },
    /// Distributed execution across multi-GPU topologies using a ring-allreduce buffer.
    // Multi-GPU support tracking individual Distributed Shards!
    MultiGpu {
        shard_id: usize,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        // Distributed Ring Topology
        peers: Vec<(Arc<wgpu::Device>, Arc<wgpu::Queue>)>,
    },
    /// MLIR (Multi-Level Intermediate Representation) JIT Execution Engine.
    /// Used for advanced compiler-fused kernel execution.
    Mlir { session_id: usize },

    /// WebGPU target for cross-browser edge computing.
    #[cfg(target_arch = "wasm32")]
    WebGpu {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    },
}

impl EngineDevice {
    /// Seamlessly extracts the GPU device and queue, whether running in Single-GPU, Multi-GPU, or Browser mode!
    pub fn get_gpu(&self) -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        match self {
            EngineDevice::Gpu { device, queue } => Some((Arc::clone(device), Arc::clone(queue))),
            EngineDevice::MultiGpu { device, queue, .. } => {
                Some((Arc::clone(device), Arc::clone(queue)))
            }
            #[cfg(target_arch = "wasm32")]
            EngineDevice::WebGpu { device, queue } => Some((Arc::clone(device), Arc::clone(queue))),
            _ => None,
        }
    }

    /// Initializes the compute device based on the available hardware environment.
    /// Defaults to high-performance GPUs, falling back to CPU if no compatible drivers are found.
    pub async fn init() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let instance = wgpu::Instance::default();

            // Attempt to find a high-performance dedicated graphics card
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    // Raw compute math mode
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await;

            match adapter {
                Ok(gpu_adapter) => {
                    let info = gpu_adapter.get_info();
                    println!(
                        ">>> Engine routing to GPU: [{}] via [{:?}] <<<",
                        info.name, info.backend
                    );

                    let (device, queue) = gpu_adapter
                        .request_device(&wgpu::DeviceDescriptor::default())
                        .await
                        .unwrap();

                    EngineDevice::Gpu {
                        device: Arc::new(device),
                        queue: Arc::new(queue),
                    }
                }
                Err(_) => {
                    // Fallback to CPU execution
                    let cores = num_cpus::get();
                    println!(
                        ">>> No GPU found. Engine routing to CPU across {} cores <<<",
                        cores
                    );

                    // Initialize the Rayon global thread pool for parallel iterators
                    let _ = rayon::ThreadPoolBuilder::new()
                        .num_threads(cores)
                        .build_global();

                    EngineDevice::Cpu { cores }
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Edge/Browser initialization using WebGPU
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU,
                ..Default::default()
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .unwrap();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap();
            EngineDevice::WebGpu {
                device: Arc::new(device),
                queue: Arc::new(queue),
            }
        }
    }

    /// Boot up the Universal MLIR Compiler
    pub fn init_mlir() -> Self {
        println!(">>> Engine routing to MLIR Universal Compiler (Tensor Cores / AMX) <<<");
        EngineDevice::Mlir { session_id: 1 }
    }
}

// Add a quick Debug implementation so we can print which GPU a tensor lives on
impl std::fmt::Debug for EngineDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineDevice::Cpu { cores } => write!(f, "CPU ({} cores)", cores),
            EngineDevice::Gpu { .. } => write!(f, "GPU (Primary)"),
            EngineDevice::MultiGpu { shard_id, .. } => write!(f, "GPU (Shard {})", shard_id),
            EngineDevice::Mlir { session_id } => {
                write!(f, "MLIR Universal JIT (Session {})", session_id)
            }
            #[cfg(target_arch = "wasm32")]
            EngineDevice::WebGpu { .. } => write!(f, "WebGPU Edge Compute"),
        }
    }
}

/// A memory-pooling manager to prevent frequent GPU allocation thrashing.
/// It recycles used buffers to minimize driver interaction.
pub struct BufferPool {
    // Caches dropped buffers by their size in bytes
    cache: Mutex<HashMap<u64, Vec<Arc<Buffer>>>>,
}

// Added Default implementation
impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferPool {
    /// Creates a new, empty buffer pool.
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Fetches a buffer of requested size. Recycles an idle one if possible,
    /// otherwise allocates a new one.
    pub fn get_buffer(
        &self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Arc<Buffer> {
        let mut cache = self.cache.lock().unwrap();

        // Collapsed the nested `if let` statements into a single line
        if let Some(buffers) = cache.get_mut(&size)
            && let Some(buf) = buffers.pop()
        {
            return buf;
        }

        // Allocate a new buffer if nothing is available in the pool
        Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pooled Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        }))
    }
}
