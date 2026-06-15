use std::sync::Arc;

#[derive(Clone)]
pub enum EngineDevice {
    Cpu { cores: usize },
    Gpu {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    },
}

impl EngineDevice {
    pub async fn init() -> Self {
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
                    &wgpu::DeviceDescriptor::default(),
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
}
