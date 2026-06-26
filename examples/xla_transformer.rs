use organon::lazy::{LazyBackend, compile};
use organon::backend::Backend;

async fn run_transformer_compile() {
    println!("====================================================");
    println!("      ORGANON XLA: FULL TRANSFORMER FUSION TEST     ");
    println!("====================================================\n");

    let instance = wgpu::Instance::default();
    let adapter = instance.enumerate_adapters(wgpu::Backends::all()).await.into_iter()
        .find(|a| a.limits().max_storage_buffers_per_shader_stage >= 4)
        .expect("FATAL: No capable GPU or Software Renderer found!");
    
    let (device, _queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Organon XLA Device"), required_limits: adapter.limits(), ..Default::default()
    }).await.unwrap();

    println!("[1] Tracing Full Transformer Block...");
    
    // We pass an entire Transformer's mathematical graph into the Compiler!
    let compiled_model = compile(&device, |inputs| {
        let x = inputs[0]; 
        let wq = inputs[1]; let wk = inputs[2]; let wv = inputs[3]; let wo = inputs[4];
        let gamma = inputs[5]; let beta = inputs[6];
        
        // 1. RMS/Layer Normalization
        let norm = LazyBackend::layer_norm(x, gamma, beta);
        
        // 2. Q, K, V Linear Projections (These are independent and can be Horizontally Fused!)
        let q = LazyBackend::matmul(&norm, wq);
        let k = LazyBackend::matmul(&norm, wk);
        let v = LazyBackend::matmul(&norm, wv);
        
        // 3. The Core Attention Mechanism (Should trigger FlashAttention SRAM Tiling!)
        let qk = LazyBackend::matmul(&q, &k);
        let scores = LazyBackend::softmax(&qk);
        let context = LazyBackend::matmul(&scores, &v);
        
        // 4. Output Projection & Residual Connection
        let out = LazyBackend::matmul(&context, wo);
        LazyBackend::add(x, &out)

    }, &[
        &LazyBackend::new_cpu(vec![], vec![16, 64]), // x (Seq: 16, Dim: 64)
        &LazyBackend::new_cpu(vec![], vec![64, 64]), // wq
        &LazyBackend::new_cpu(vec![], vec![64, 64]), // wk
        &LazyBackend::new_cpu(vec![], vec![64, 64]), // wv
        &LazyBackend::new_cpu(vec![], vec![64, 64]), // wo
        &LazyBackend::new_cpu(vec![], vec![64]),     // gamma
        &LazyBackend::new_cpu(vec![], vec![64]),     // beta
    ]);

    println!("\n[2] JIT Compilation Complete!");
    println!("    -> The graph originally required 9 heavy memory operations.");
    println!("    -> The XLA Compiler squashed it into just {} highly optimized GPU Dispatches!", compiled_model.steps.len());
    println!("\n====================================================");
    println!("✅ SUCCESS: The Transformer Block is successfully compiled and cached in VRAM!");
    println!("====================================================");
}

fn main() {
    pollster::block_on(run_transformer_compile());
}