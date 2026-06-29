/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Rakesh Pradip Dey
 *
 * Licensed under the MIT License <LICENSE-MIT or http://opensource.org/licenses/MIT>.
 *
 * Note: Portions of this software are adapted from existing open-source frameworks.
 * This file may not be copied, modified, or distributed except according to the terms
 * of the MIT license.
 */

use clove::backend::WgpuBackend;
use std::time::Instant;

/// This benchmark measures the time taken for a matrix multiplication
/// of size 1024x1024.
fn main() {
    let size = 1024;

    // 1. Setup random data
    let data_a: Vec<f32> = (0..size * size).map(|_| rand::random()).collect();
    let data_b: Vec<f32> = (0..size * size).map(|_| rand::random()).collect();

    // 2. Perform MatMul in Clove (Rayon Backend)
    println!("Benchmarking Clove MatMul (1024x1024)...");

    let start = Instant::now();
    let _result = WgpuBackend::rayon_matmul(&data_a, &data_b, size, size, size);
    let duration = start.elapsed();

    println!("Clove: {:.2} ms", duration.as_millis());
}
