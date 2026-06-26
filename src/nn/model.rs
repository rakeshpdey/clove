use crate::nn::{Linear, TransformerBlock, Embedding};
use crate::tensor::TensorNode;
use crate::backend::Backend;
use ndarray::Array2;

use std::fs::File;
use std::io::Result;
use crate::backend::{WgpuBackend, TensorData};
use crate::device::EngineDevice;
use std::collections::HashMap;
use safetensors::tensor::{SafeTensors, TensorView, Dtype};

pub struct LanguageModel<B: Backend> {
    pub embed: Embedding<B>,
    pub blocks: Vec<TransformerBlock<B>>,
    pub head: Linear<B>,
}

impl<B: Backend> LanguageModel<B> {
    pub fn new(vocab_size: usize, hidden_dim: usize, num_layers: usize, num_heads: usize) -> Self {
        let mut blocks = Vec::new();
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(hidden_dim, num_heads));
        }

        Self {
            embed: Embedding::new(vocab_size, hidden_dim),
            blocks,
            head: Linear::new(hidden_dim, vocab_size),
        }
    }

    pub fn forward(&self, indices: &Array2<f32>) -> TensorNode<B> {
        let mut x = self.embed.forward(indices);

        for block in &self.blocks {
            x = block.forward(&x);
        }
        
        self.head.forward(&x)
    }

    pub fn parameters(&self) -> Vec<TensorNode<B>> {
        let mut params = self.embed.parameters();
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.head.parameters());
        params
    }
}

// ========================================================================
// STANDARDIZED SERIALIZATION (SAFETENSORS)
// ========================================================================
impl LanguageModel<WgpuBackend> {
    /// Saves all model parameters into a standard Safetensors file using sequential indexing.
    pub fn save_safetensors(&self, path: &str) -> Result<()> {
        let params = self.parameters();
        let mut views = HashMap::new();
        let mut cpu_tensors = Vec::new();

        // 1. Collect and pull all nodes to the CPU sequentially
        for (i, node) in params.iter().enumerate() {
            let tensor = node.read().unwrap();
            let (data_vec, _) = tensor.to_cpu().into_raw_vec_and_offset();
            // Generate a deterministic structural name based on position
            cpu_tensors.push((format!("parameter.{}", i), data_vec, tensor.shape.clone()));
        }

        // 2. Wrap raw binary slices into Safetensors views
        for (name, data, shape) in &cpu_tensors {
            let byte_data: &[u8] = bytemuck::cast_slice(data);
            let view = TensorView::new(Dtype::F32, shape.clone(), byte_data).unwrap();
            views.insert(name.clone(), view);
        }

        // 3. Write standard header data and payloads out to the filesystem
        safetensors::tensor::serialize_to_file(&views, None, std::path::Path::new(path))
            .expect("Failed to serialize Safetensors");
            
        println!("Successfully saved Safetensors checkpoint to: {}", path);
        Ok(())
    }

    /// Zero-Copy loads a Safetensors model from disk directly into Framework Memory.
    pub fn load_safetensors(&self, path: &str) -> Result<()> {
        // 1. Memory-map the target file directly into memory space
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        
        // 2. Instantly parse out the validation header map
        let tensors = SafeTensors::deserialize(&mmap)
            .expect("Failed to parse Safetensors header. Is the file corrupted?");
            
        let params = self.parameters();

        // 3. Line up the file parameters with your exact tracked graph sequence
        for (i, node) in params.iter().enumerate() {
            let name = format!("parameter.{}", i);
            
            if let Ok(tensor_view) = tensors.tensor(&name) {
                if tensor_view.dtype() != Dtype::F32 {
                    panic!("Tensor {} is not F32. Your framework currently only supports full precision loading.", name);
                }

                let float_data: Vec<f32> = bytemuck::cast_slice(tensor_view.data()).to_vec();
                let mut param = node.write().unwrap();
                
                // Keep structural protection!
                assert_eq!(param.shape, tensor_view.shape(), 
                    "Shape mismatch on parameter position {}! Expected {:?}, got {:?}", i, param.shape, tensor_view.shape());

                let device = param.device.clone();

                match &mut param.data {
                    TensorData::Cpu(d) => *d = float_data,
                    TensorData::Gpu(buf) => {
                        if let EngineDevice::Gpu { queue, .. } = &device {
                            queue.write_buffer(buf, 0, bytemuck::cast_slice(&float_data));
                        }
                    }
                    TensorData::Lazy(_) => panic!("Cannot push bytes to Lazy MLIR node"),
                }
            } else {
                println!("⚠️ Warning: Parameter indexing slot '{}' was not located in the Safetensors checkpoint.", name);
            }
        }

        println!("Successfully loaded Safetensors into network from: {}", path);
        Ok(())
    }
}