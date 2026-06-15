struct SgdArgs {
    lr: f32,
    clip: f32,
    size: u32,
}

@group(0) @binding(0) var<uniform> args: SgdArgs;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read> gradients: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Safety boundary: don't read past the end of the parameter vector
    if (idx >= args.size) {
        return;
    }

    // 1. Fetch the raw gradient computed by backpropagation
    let raw_grad = gradients[idx];

    // 2. Apply explicit gradient clipping brakes directly on the GPU core
    let clipped_grad = clamp(raw_grad, -args.clip, args.clip);

    // 3. Mathematical Formula: Weight = Weight - (Learning_Rate * Clipped_Gradient)
    weights[idx] = weights[idx] - (args.lr * clipped_grad);
}
