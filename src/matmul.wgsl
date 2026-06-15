// 1. We define the matrix dimensions (M, K, N)
struct Dimensions {
    m: u32,
    k: u32,
    n: u32,
}

// 2. We bind the memory slots. These point directly to the VRAM buffers.
@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> matrixA: array<f32>;
@group(0) @binding(2) var<storage, read> matrixB: array<f32>;
@group(0) @binding(3) var<storage, read_write> matrixC: array<f32>;

// 3. The Thread Grid Configuration
// We tell the GPU to organize its threads into 8x8 blocks
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    
    // The GPU gives every single thread a unique X/Y coordinate
    let row = global_id.y;
    let col = global_id.x;

    // Safety check: If the thread falls outside our matrix dimensions, kill it immediately
    if (row >= dims.m || col >= dims.n) {
        return;
    }

    // 4. The Dot Product Loop
    // This single thread calculates exactly ONE number for the final output matrix
    var sum: f32 = 0.0;
    for (var p: u32 = 0u; p < dims.k; p = p + 1u) {
        // Find the correct elements in the 1D arrays acting as 2D matrices
        let a_val = matrixA[row * dims.k + p];
        let b_val = matrixB[p * dims.n + col];
        sum = sum + (a_val * b_val);
    }
    
    // Save the computed result directly back to VRAM
    matrixC[row * dims.n + col] = sum;
}
