/*
 * src/distributed.rs
 * Multi-GPU Orchestrator
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

//! Distributed Multi-GPU training orchestration and networking.
//!
//! This module implements a high-performance Ring-AllReduce collective
//! communication protocol, allowing multiple GPU shards to synchronize
//! gradients across local and network-distributed nodes.
//!
//! NOTE: This module requires the "full" feature flag for tokio in your Cargo.toml:

use crate::device::EngineDevice;
use rayon::prelude::*;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use wgpu::{Device, Queue};

/// Represents a single physical GPU worker node on the CURRENT machine.
pub struct GPUShard {
    pub shard_id: usize,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub engine_device: EngineDevice,
}

/// The Orchestrator manages the pool of local GPUs AND Cross-Node Network Sync!
///
/// It utilizes an embedded Tokio runtime to bridge asynchronous network operations
/// into the synchronous autograd/training loop.
pub struct MultiGPUOrchestrator {
    pub shards: Vec<GPUShard>,

    // True Ring-AllReduce Network Topology (NCCL Architecture) ---
    pub rank: usize,
    pub world_size: usize,

    // In a Ring Topology, we strictly hold one connection from the left, and one to the right.
    pub rx_stream: Option<Arc<tokio::sync::Mutex<tokio::net::TcpStream>>>,
    pub tx_stream: Option<Arc<tokio::sync::Mutex<tokio::net::TcpStream>>>,

    // Embedded Tokio runtime to bridge our asynchronous network inside the synchronous autograd loop
    pub rt: Arc<tokio::runtime::Runtime>,
}

impl MultiGPUOrchestrator {
    /// Initializes all available physical adapters on the host system.
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;

        let mut shards = Vec::new();
        let mut shard_id = 0;

        for adapter in adapters {
            let limits = adapter.limits();
            if limits.max_compute_workgroups_per_dimension == 0 {
                continue;
            }

            println!(
                "[ORCHESTRATOR] Found Capable GPU Shard {}: {:?}",
                shard_id,
                adapter.get_info().name
            );

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some(&format!("GPU Shard {}", shard_id)),
                    required_limits: limits,
                    ..Default::default()
                })
                .await
                .expect("FATAL: Failed to create device connection for shard.");

            let dev_arc = Arc::new(device);
            let q_arc = Arc::new(queue);

            shards.push(GPUShard {
                shard_id,
                device: Arc::clone(&dev_arc),
                queue: Arc::clone(&q_arc),
                engine_device: EngineDevice::MultiGpu {
                    shard_id,
                    device: dev_arc,
                    queue: q_arc,
                    peers: vec![],
                },
            });

            shard_id += 1;
        }

        println!(
            "[ORCHESTRATOR] Successfully initialized {} Local GPU Shards.",
            shards.len()
        );

        // Initialize the embedded Tokio Runtime for the sync bridge
        let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());

        MultiGPUOrchestrator {
            shards,
            rank: 0,
            world_size: 1, // Defaults to a local 1-node cluster
            rx_stream: None,
            tx_stream: None,
            rt,
        }
    }

    /// THE RING-ALLREDUCE CLUSTER HANDSHAKE
    /// Binds multiple physical computers together in a flawless Ring Topology!
    pub async fn connect_cluster(&mut self, my_rank: usize, all_addrs: Vec<String>) {
        self.world_size = all_addrs.len();
        self.rank = my_rank;

        if self.world_size <= 1 {
            println!("[NETWORK] Running in Local Mode (World Size: 1).");
            return;
        }

        let my_addr = &all_addrs[my_rank];
        // Calculate the neighbor to the right (wraps around to 0 at the end of the ring)
        let next_addr = &all_addrs[(my_rank + 1) % self.world_size];

        println!("[NETWORK] Node Rank {} booting on {}.", my_rank, my_addr);
        println!("[NETWORK] Target Right-Neighbor: {}", next_addr);

        // Open a listener for the Left-Neighbor
        let listener = tokio::net::TcpListener::bind(my_addr)
            .await
            .expect("FATAL: Failed to bind to local port! Is the port already in use?");

        // Dial the Right-Neighbor
        // We use a retry loop so you can boot the computers in any order.
        let mut tx_stream_opt = None;
        for attempt in 1..=30 {
            if let Ok(stream) = tokio::net::TcpStream::connect(next_addr).await {
                println!(
                    "[NETWORK] Rank {} successfully connected to Right-Neighbor!",
                    my_rank
                );
                tx_stream_opt = Some(stream);
                break;
            }
            println!(
                "[NETWORK] Waiting for Right-Neighbor to come online... (Attempt {}/30)",
                attempt
            );
            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
        }
        let tx_stream =
            tx_stream_opt.expect("FATAL: Failed to connect to the next node in the ring!");

        // Accept connection from the Left-Neighbor
        let (rx_stream, prev_addr) = listener
            .accept()
            .await
            .expect("FATAL: Failed to accept connection from Left-Neighbor!");

        println!(
            "[NETWORK] Rank {} accepted connection from Left-Neighbor at {}!",
            my_rank, prev_addr
        );

        self.rx_stream = Some(Arc::new(tokio::sync::Mutex::new(rx_stream)));
        self.tx_stream = Some(Arc::new(tokio::sync::Mutex::new(tx_stream)));

        println!("[NETWORK] Ring-AllReduce Cluster is Fully Synchronized and Ready!");
    }

    /// Distributes a global batch size perfectly across all available LOCAL GPUs.
    pub fn distribute_batch(&self, total_batch_size: usize) -> Vec<usize> {
        let num_shards = self.shards.len();
        if num_shards == 0 {
            return vec![total_batch_size];
        }
        let base_size = total_batch_size / num_shards;
        let remainder = total_batch_size % num_shards;
        (0..num_shards)
            .map(|i| {
                if i < remainder {
                    base_size + 1
                } else {
                    base_size
                }
            })
            .collect()
    }

    /// THE GLOBAL ALL-REDUCE MEMORY BRIDGE
    /// Synchronizes gradients across local GPUs AND TCP/IP Network streams!
    pub fn synchronize_gradients(&self, buffers: &[Arc<wgpu::Buffer>]) {
        let num_shards = self.shards.len();

        // If there's only 1 local GPU and no network cluster, math sync is unneeded
        if (num_shards <= 1 && self.world_size <= 1) || buffers.is_empty() {
            return;
        }

        let buffer_size = buffers[0].size();
        let num_elements = (buffer_size / 4) as usize;
        let mut all_data = Vec::with_capacity(num_shards);

        // PULL LOCAL (Read data from all local GPU VRAMs into CPU RAM)
        // We poll the GPU until staging buffers are populated.
        for (i, shard) in self.shards.iter().enumerate() {
            let staging_buf = shard.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("AllReduce Staging Read"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = shard
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(&buffers[i], 0, &staging_buf, 0, buffer_size);
            shard.queue.submit(std::iter::once(encoder.finish()));

            let slice = staging_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
            shard
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            rx.recv().unwrap().unwrap();

            let mapped = slice.get_mapped_range();
            let floats: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
            all_data.push(floats);
        }

        // AVERAGE LOCAL (Use Rayon to multi-thread the local CPU math)
        // Reduces gradients locally first to optimize bandwidth usage before crossing the wire.
        let mut local_averaged = vec![0.0f32; num_elements];
        let local_scale = 1.0 / (num_shards as f32);

        local_averaged
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, avg_val)| {
                let mut sum = 0.0;
                for shard in all_data.iter().take(num_shards) {
                    sum += shard[idx];
                }
                *avg_val = sum * local_scale;
            });

        // THE GLOBAL TCP/IP RING-ALLREDUCE (True NCCL Architecture!)
        let global_averaged = if self.world_size > 1 {
            // We use `block_on` so the synchronous training loop waits for the network packet!
            self.rt.block_on(async {
                let mut rx = self.rx_stream.as_ref().unwrap().lock().await;
                let mut tx = self.tx_stream.as_ref().unwrap().lock().await;

                let mut global_data = local_averaged.clone();
                let mut pass_along = local_averaged.clone();
                let mut recv_buffer = vec![0.0f32; num_elements];

                // Ring Accumulation: Data gets passed around the ring N-1 times.
                // By the end, every single node has received the gradient slice from every other node!
                for _ in 0..(self.world_size - 1) {
                    // tokio::try_join! executes the network send and receive completely simultaneously!
                    // This prevents deadlocks and maximizes network bandwidth throughput.
                    let send_future = tx.write_all(bytemuck::cast_slice(&pass_along));
                    let recv_future = rx.read_exact(bytemuck::cast_slice_mut(&mut recv_buffer));

                    tokio::try_join!(send_future, recv_future)
                        .expect("FATAL: Ring connection broken during All-Reduce!");

                    // Add the data we just received to our global total
                    global_data
                        .par_iter_mut()
                        .zip(recv_buffer.par_iter())
                        .for_each(|(g, r)| {
                            *g += r;
                        });

                    // The data we pass along in the NEXT step is the data we just received
                    pass_along.copy_from_slice(&recv_buffer);
                }

                // Average across all nodes in the world
                let global_scale = 1.0 / (self.world_size as f32);
                global_data.par_iter_mut().for_each(|val| {
                    *val *= global_scale;
                });

                global_data
            })
        } else {
            local_averaged
        };

        // PUSH (Blast the perfect global gradients back to every local GPU!)
        let bytes_to_write = bytemuck::cast_slice(&global_averaged);
        for (i, shard) in self.shards.iter().enumerate() {
            shard.queue.write_buffer(&buffers[i], 0, bytes_to_write);
        }
    }
}
