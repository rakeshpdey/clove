// We replicate the core rotation logic here to test it in isolation
fn test_rotate(x: f32, y: f32, theta: f32) -> (f32, f32) {
    let x_rot = x * theta.cos() - y * theta.sin();
    let y_rot = x * theta.sin() + y * theta.cos();
    (x_rot, y_rot)
}

fn main() {
    println!("--- RoPE Mathematical Validation ---");

    // 1. Test: Identity (0.0 rotation)
    let (x1, y1) = test_rotate(1.0, 0.0, 0.0);
    println!("Test Identity (0 rad): Expected [1.0, 0.0], Got [{:.4}, {:.4}]", x1, y1);
    assert!((x1 - 1.0).abs() < 1e-5 && y1.abs() < 1e-5);

    // 2. Test: 90 degrees (pi/2 rotation)
    let pi_2 = std::f32::consts::FRAC_PI_2;
    let (x2, y2) = test_rotate(1.0, 0.0, pi_2);
    println!("Test 90deg (pi/2 rad): Expected [0.0, 1.0], Got [{:.4}, {:.4}]", x2, y2);
    assert!(x2.abs() < 1e-5 && (y2 - 1.0).abs() < 1e-5);

    // 3. Test: 180 degrees (pi rotation)
    let pi = std::f32::consts::PI;
    let (x3, y3) = test_rotate(1.0, 0.0, pi);
    println!("Test 180deg (pi rad):  Expected [-1.0, 0.0], Got [{:.4}, {:.4}]", x3, y3);
    assert!((x3 + 1.0).abs() < 1e-5 && y3.abs() < 1e-5);

    println!("\nSUCCESS: RoPE rotation math is verified and ready for integration!");
}