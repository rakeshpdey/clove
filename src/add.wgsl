struct Dimensions {
    a_size: u32,
    b_size: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> arrayA: array<f32>;
@group(0) @binding(2) var<storage, read> arrayB: array<f32>;
@group(0) @binding(3) var<storage, read_write> arrayC: array<f32>;

// Group threads into blocks of 256 for rapid 1D array processing
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Safety boundary
    if (idx >= dims.a_size) {
        return;
    }

    // The modulo (%) handles broadcasting automatically!
    arrayC[idx] = arrayA[idx] + arrayB[idx % dims.b_size];
}
