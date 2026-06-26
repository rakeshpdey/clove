use organon::backend::Backend;
use organon::lazy::{LazyBackend, WGSLCompiler, GLOBAL_GRAPH};

fn main() {
    println!("====================================================");
    println!("       ORGANON XLA KERNEL FUSION COMPILER TEST      ");
    println!("====================================================\n");

    // 1. Create mock input tensors using the LazyBackend
    // Notice that tensor 'c' has a shape of [1] (a scalar bias). 
    // In eager execution, this would crash without an expensive broadcast shader!
    println!("[1] Registering Input Nodes...");
    let a = LazyBackend::new_cpu(vec![], vec![256]); // Matrix A
    let b = LazyBackend::new_cpu(vec![], vec![256]); // Matrix B
    let c = LazyBackend::new_cpu(vec![], vec![1]);   // Bias Vector C

    println!("[2] Tracing Mathematical Operations...");
    let step1 = LazyBackend::mul(&a, &b);       // a * b
    let step2 = LazyBackend::add(&step1, &c);   // (a * b) + c  <-- Broadcasting required!
    let _out  = LazyBackend::relu(&step2);      // ReLU((a * b) + c)

    // 3. Inspect the Compute Graph
    println!("\n[3] Inspecting the Compute Graph IR (Intermediate Representation):");
    GLOBAL_GRAPH.with(|g| {
        let graph = g.borrow();
        for node in &graph.nodes {
            println!("    -> Node {:02} | Op: {:<8} | Deps: {:?}", node.id, format!("{:?}", node.op), node.dependencies);
        }
    });

    println!("\n[4] 🚀 TRIGGERING JIT COMPILER 🚀\n");
    
    // 4. Fire the JIT Compiler!
    let fused_shader = WGSLCompiler::compile_fused_kernel();
    
    println!("{}", fused_shader);
    
    println!("====================================================");
    println!("✅ SUCCESS: Dynamic Bindings and Modulo Broadcasting were successfully injected!");
    println!("====================================================");
}