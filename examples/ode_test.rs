use organon::tensor::{Tensor, TensorData, Node};
use organon::device::EngineDevice;
use organon::AdamW;
use organon::nn::NeuralODE;
use std::sync::{Arc, RwLock};

// Simple MSE Loss for our 2D coordinate targets
fn mse_loss(pred: &Node, target: &Node) -> Node {
    let p_read = pred.read().unwrap();
    let t_read = target.read().unwrap();
    let p_data = if let TensorData::Cpu(d) = &p_read.data { d } else { panic!("MSE needs CPU") };
    let t_data = if let TensorData::Cpu(d) = &t_read.data { d } else { panic!("MSE needs CPU") };
    
    let mut loss_val = 0.0;
    let mut grad = vec![0.0; p_data.len()];
    let n = p_data.len() as f32;
    for i in 0..p_data.len() {
        let diff = p_data[i] - t_data[i];
        loss_val += diff * diff;
        grad[i] = (2.0 * diff) / n;
    }
    loss_val /= n;
    
    let pred_clone = Arc::clone(pred);
    Arc::new(RwLock::new(Tensor {
        data: TensorData::Cpu(vec![loss_val]),
        shape: vec![1],
        grad: None,
        creators: vec![Arc::clone(pred)],
        device: EngineDevice::Cpu { cores: 1 },
        backward: Some(Box::new(move |out_tensor: &Tensor| {
            let out_grad = out_tensor.get_cpu_grad()[0];
            let mut final_grad = grad.clone();
            for val in final_grad.iter_mut() { *val *= out_grad; }
            pred_clone.write().unwrap().add_cpu_grad(&final_grad);
        }))
    }))
}

// =====================================================================
// THE AUTOGRAD ENGINE
// Maps the entire physics simulation backwards to apply gradients!
// =====================================================================
fn backward_graph(root: &Node) {
    let mut topo = Vec::new();
    let mut visited = std::collections::HashSet::new();

    // 1. Recursively build the computational graph
    fn build_topo(node: &Node, topo: &mut Vec<Node>, visited: &mut std::collections::HashSet<usize>) {
        let ptr = Arc::as_ptr(node) as usize;
        if !visited.contains(&ptr) {
            visited.insert(ptr);
            // Clone creators to prevent deadlocks while traversing
            let creators = node.read().unwrap().creators.clone();
            for child in &creators {
                build_topo(child, topo, visited);
            }
            topo.push(Arc::clone(node));
        }
    }

    build_topo(root, &mut topo, &mut visited);

    // 2. Traverse backwards and apply the calculus chain rule
    for node in topo.iter().rev() {
        let tensor = node.read().unwrap();
        if let Some(back_fn) = &tensor.backward {
            back_fn(&tensor);
        }
    }
}

fn main() {
    println!("--- ORGANON: Continuous-Time Neural ODE ---");

    // We use a 2D dimension. The ODE will take 20 physics steps to reach the end.
    let dim = 2;
    let ode_block = NeuralODE::new(dim, 20); 
    let mut optimizer = AdamW::new(0.01, ode_block.parameters());

    // We want to bend the fabric of space so that [1.0, 0.0] flows into [0.0, 1.0]
    let input_point = vec![1.0, 0.0];
    let target_point = vec![0.0, 1.0];

    let x_node = Arc::new(RwLock::new(Tensor { 
        data: TensorData::Cpu(input_point.clone()), 
        shape: vec![1, dim], grad: None, creators: vec![], device: EngineDevice::Cpu { cores: 1 }, backward: None 
    }));
    let y_node = Arc::new(RwLock::new(Tensor { 
        data: TensorData::Cpu(target_point.clone()), 
        shape: vec![1, dim], grad: None, creators: vec![], device: EngineDevice::Cpu { cores: 1 }, backward: None 
    }));

    println!("Initial Point: {:?}", input_point);
    println!("Target Point: {:?}", target_point);
    println!("Simulating Flow...");

    for epoch in 1..=500 {
        let out_node = ode_block.forward(&x_node);
        let loss = mse_loss(&out_node, &y_node);
        
        let loss_val = if let TensorData::Cpu(d) = &loss.read().unwrap().data { d[0] } else { 0.0 };
        
        optimizer.zero_grad();
        loss.write().unwrap().grad = Some(TensorData::Cpu(vec![1.0]));
        
        // Use our new Autograd Engine to ripple the math through time!
        backward_graph(&loss);
        
        optimizer.step();

        if epoch % 50 == 0 {
            // Let's print where the point is currently landing!
            let current_pos = if let TensorData::Cpu(d) = &out_node.read().unwrap().data { d.clone() } else { vec![] };
            println!("Epoch: {} | Loss: {:.6} | Point landed at: [{:.4}, {:.4}]", epoch, loss_val, current_pos[0], current_pos[1]);
        }
    }
    println!("Physics Simulation Complete!");
}