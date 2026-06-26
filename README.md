# Organon

Organon is a high-performance machine learning framework. Designed for both scale-out cluster training and edge-native inference, it leverages hardware acceleration via wgpu, lazy evaluation for kernel fusion, and a robust tape-based autograd engine.

## Core Architecture

* **Lazy Execution & JIT Compilation:** Organon dynamically traces computation graphs into an Intermediate Representation (IR). The LazyEngine performs dead-code elimination, constant folding, and horizontal kernel fusion before compiling the graph into hyper-optimized WGSL shaders.
* **Hardware Agnostic (CPU/GPU/WASM):** A unified Backend trait abstracts hardware complexities. Run seamlessly on multi-core CPUs via Rayon, dedicated GPUs via Vulkan/Metal/DX12, or directly in the browser using WebAssembly and WebGPU.
* **Advanced LLM Meta:** Native implementation of PagedAttention for zero-fragmentation KV-cache memory management, enabling high-throughput inference for Transformer-based architectures.
* **Production Training Suite:** Includes a sophisticated Optim module featuring AdamW, learning rate schedulers (CosineAnnealingLR), and a dynamic GradScaler for safe Automatic Mixed Precision (AMP) training.
* **Distributed Native:** First-class support for Multi-GPU training topologies using Ring-AllReduce collective communication paradigms.

## Ecosystem Interoperability

Organon is designed to integrate into existing ML infrastructure, not isolate itself:
* **C-ABI / FFI:** Exposes a safe C Application Binary Interface (ffi.rs), allowing Organon to be driven as a high-performance backend for other Languages.
* **ONNX Export:** Built-in Protobuf visitor (ONNXExporter) allows any Organon computation graph to be instantly exported to .onnx for deployment to TensorRT or CoreML.

## Prerequisites

* **Rust:** Latest stable toolchain (install via rustup).
* **Hardware:** Vulkan, Metal, or DX12 compliant drivers for GPU acceleration.

## Installation & Build

Clone the repository and compile the highly-optimized release build:

\`\`\`bash
git clone https://github.com/rakeshpdey/organon
cd organon
cargo build --release
\`\`\`

## Contributing

We welcome community contributions. Please ensure that all new operations include corresponding WGSL shader implementations in backend.rs and appropriate test coverage.

## License

Organon is distributed under the terms of the MIT license.
