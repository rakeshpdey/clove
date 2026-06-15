struct Dimensions {
    a_size: u32,
    b_size: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> grad_out: array<f32>;       // The incoming error signal (dC)
@group(0) @binding(2) var<storage, read_write> grad_a: array<f32>;   // The outgoing error for A
@group(0) @binding(3) var<storage, read_write> grad_b: array<atomic<u32>>; // Atomic memory for B!

// Helper function to safely add floats together across thousands of colliding threads
fn atomicAddFloat(index: u32, value: f32) {
    var expected: u32 = atomicLoad(&grad_b[index]);
    loop {
        // 1. Read the current memory as a float
        let current_f32: f32 = bitcast<f32>(expected);
        // 2. Add our thread's gradient to it
        let next_f32: f32 = current_f32 + value;
        // 3. Convert it back to raw bytes (u32) for the memory lock
        let next_u32: u32 = bitcast<u32>(next_f32);
        
        // 4. Try to write it. If another thread snuck in and changed the memory 
        // while we were doing the math, it fails, and we loop again!
        let exchange_result = atomicCompareExchangeWeak(&grad_b[index], expected, next_u32);
        if (exchange_result.exchanged) {
            break;
        }
        expected = exchange_result.old_value;
    }
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= dims.a_size) {
        return;
    }

    // Grab the error signal from the parent
    let g = grad_out[idx];

    // 1. Matrix A is 1-to-1 mapped. No thread collisions. We can write directly.
    grad_a[idx] = grad_a[idx] + g;

    // 2. Matrix B might be smaller (broadcasted). We MUST use the atomic lock.
    atomicAddFloat(idx % dims.b_size, g);
}