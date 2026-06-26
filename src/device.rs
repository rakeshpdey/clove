use std::sync::Arc;

// Conditionally bring in WASM requirements when compiling for the browser!
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[derive(Clone)]
pub enum EngineDevice {
    Cpu { cores: usize },
    Gpu {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    },
    // Multi-GPU support tracking individual Distributed Shards!
    MultiGpu { 
        shard_id: usize, 
        device: Arc<wgpu::Device>, 
        queue: Arc<wgpu::Queue>,
        peers: Vec<(Arc<wgpu::Device>, Arc<wgpu::Queue>)>, // NEW: Distributed Ring Topology
    },
    // NEW: MLIR JIT Execution Engine (Hardware Agnostic Tensor Cores)
    Mlir {
        // In a production melior-crate setup, this holds the MLIR Context & ExecutionEngine
        session_id: usize,
    },
    
    // NEW: WebAssembly WebGPU Target
    #[cfg(target_arch = "wasm32")]
    WebGpu {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    }
}

impl EngineDevice {
    /// Seamlessly extracts the GPU device and queue, whether running in Single-GPU, Multi-GPU, or Browser mode!
    pub fn get_gpu(&self) -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        match self {
            EngineDevice::Gpu { device, queue } => Some((Arc::clone(device), Arc::clone(queue))),
            EngineDevice::MultiGpu { device, queue, .. } => Some((Arc::clone(device), Arc::clone(queue))),
            #[cfg(target_arch = "wasm32")]
            EngineDevice::WebGpu { device, queue } => Some((Arc::clone(device), Arc::clone(queue))),
            _ => None,
        }
    }

    pub async fn init() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let instance = wgpu::Instance::default();
            
            // Attempt to find a high-performance dedicated graphics card
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // Raw compute math mode
                force_fallback_adapter: false,
            }).await;

            match adapter {
                Ok(gpu_adapter) => {
                    let info = gpu_adapter.get_info();
                    println!(">>> Engine routing to GPU: [{}] via [{:?}] <<<", info.name, info.backend);

                    let (device, queue) = gpu_adapter.request_device(
                        &wgpu::DeviceDescriptor::default()
                    ).await.unwrap();

                    EngineDevice::Gpu {
                        device: Arc::new(device),
                        queue: Arc::new(queue),
                    }
                }
                Err(_) => {
                    let cores = num_cpus::get();
                    println!(">>> No GPU found. Engine routing to CPU across {} cores <<<", cores);
                    
                    // Initialize the Rayon global thread pool for parallel iterators
                    let _ = rayon::ThreadPoolBuilder::new()
                        .num_threads(cores)
                        .build_global();

                    EngineDevice::Cpu { cores }
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Edge/Browser initialization using WebGPU
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU,
                ..Default::default()
            });
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
            let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.unwrap();
            EngineDevice::WebGpu { device: Arc::new(device), queue: Arc::new(queue) }
        }
    }

    /// Boot up the Universal MLIR Compiler
    pub fn init_mlir() -> Self {
        println!(">>> Engine routing to MLIR Universal Compiler (Tensor Cores / AMX) <<<");
        EngineDevice::Mlir { session_id: 1 }
    }
}

// Add a quick Debug implementation so we can print which GPU a tensor lives on
impl std::fmt::Debug for EngineDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineDevice::Cpu { cores } => write!(f, "CPU ({} cores)", cores),
            EngineDevice::Gpu { .. } => write!(f, "GPU (Primary)"),
            EngineDevice::MultiGpu { shard_id, .. } => write!(f, "GPU (Shard {})", shard_id),
            EngineDevice::Mlir { session_id } => write!(f, "MLIR Universal JIT (Session {})", session_id),
            #[cfg(target_arch = "wasm32")]
            EngineDevice::WebGpu { .. } => write!(f, "WebGPU Edge Compute"),
        }
    }
}