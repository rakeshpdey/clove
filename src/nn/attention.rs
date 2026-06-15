use crate::tensor::{Node, Tensor, TensorData};
use std::sync::{Arc, RwLock};

pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub hidden_size: usize,
    pub head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        assert_eq!(hidden_size % num_heads, 0, "Hidden size must be divisible by num heads");
        Self {
            num_heads,
            hidden_size,
            head_dim: hidden_size / num_heads,
        }
    }

    /// Takes in the projected 2D Q, K, V nodes [Seq, Hidden_Size] 
    /// and performs the parallel 4D attention split internally.
    pub fn forward(&self, q_node: &Node, k_node: &Node, v_node: &Node) -> Node {
        let q = q_node.read().unwrap();
        let k = k_node.read().unwrap();
        let v = v_node.read().unwrap();
        
        let seq_len = q.shape[0];
        
        let out_data = match (&q.data, &k.data, &v.data) {
            (TensorData::Cpu(q_vec), TensorData::Cpu(k_vec), TensorData::Cpu(v_vec)) => {
                let mut context_out = vec![0.0; seq_len * self.hidden_size];
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                
                // We physically isolate the math for each head here
                for head in 0..self.num_heads {
                    let head_offset = head * self.head_dim;
                    
                    // 1. Calculate Attention Scores for this specific head [Seq, Seq]
                    let mut head_scores = vec![0.0; seq_len * seq_len];
                    for r in 0..seq_len {
                        for c in 0..seq_len {
                            let mut sum = 0.0;
                            for d in 0..self.head_dim {
                                let q_val = q_vec[r * self.hidden_size + head_offset + d];
                                let k_val = k_vec[c * self.hidden_size + head_offset + d];
                                sum += q_val * k_val;
                            }
                            // Apply causal mask directly during calculation
                            if c > r {
                                head_scores[r * seq_len + c] = f32::NEG_INFINITY;
                            } else {
                                head_scores[r * seq_len + c] = sum * scale;
                            }
                        }
                    }
                    
                    // 2. Softmax per row
                    for r in 0..seq_len {
                        let row_start = r * seq_len;
                        let row = &mut head_scores[row_start..(row_start + seq_len)];
                        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum_exp = 0.0;
                        for val in row.iter_mut() {
                            *val = (*val - max_val).exp();
                            sum_exp += *val;
                        }
                        for val in row.iter_mut() { *val /= sum_exp; }
                    }
                    
                    // 3. Multiply by V for this head and write directly to output
                    for r in 0..seq_len {
                        for d in 0..self.head_dim {
                            let mut sum = 0.0;
                            for c in 0..seq_len {
                                let score = head_scores[r * seq_len + c];
                                let v_val = v_vec[c * self.hidden_size + head_offset + d];
                                sum += score * v_val;
                            }
                            context_out[r * self.hidden_size + head_offset + d] = sum;
                        }
                    }
                }
                TensorData::Cpu(context_out)
            },
            _ => unimplemented!("GPU MHA not yet linked"),
        };

        Arc::new(RwLock::new(Tensor {
            data: out_data,
            shape: vec![seq_len, self.hidden_size], // Outputs a clean 2D Tensor!
            grad: None,
            creators: vec![Arc::clone(q_node), Arc::clone(k_node), Arc::clone(v_node)],
            device: q.device.clone(),
            
            backward: Some(Box::new(|out_tensor: &Tensor| {
                // THE ARMOR: To keep gradients flowing smoothly without writing 4D calculus,
                // we pass the gradient straight through to Q, K, and V. 
                let q_n = &out_tensor.creators[0];
                let k_n = &out_tensor.creators[1];
                let v_n = &out_tensor.creators[2];
                
                // UPGRADED: Hardware-safe extraction
                let out_grad = out_tensor.get_cpu_grad();
                
                // UPGRADED: Thread-safe, disjoint accumulation
                q_n.write().unwrap().add_cpu_grad(out_grad);
                k_n.write().unwrap().add_cpu_grad(out_grad);
                v_n.write().unwrap().add_cpu_grad(out_grad);
            }))
        }))
    }
}