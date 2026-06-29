# Clove

Clove is a high-performance machine learning framework. Designed for both scale-out cluster training and edge-native inference, it leverages hardware acceleration via wgpu, lazy evaluation for kernel fusion, and a robust tape-based autograd engine.

## Core Architecture

* **Lazy Execution & JIT Compilation:** Clove dynamically traces computation graphs into an Intermediate Representation. The LazyEngine performs dead-code elimination, constant folding, and horizontal kernel fusion before compiling the graph into hyper-optimized WGSL shaders.
* **Hardware Agnostic (CPU/GPU/WASM):** A unified Backend trait abstracts hardware complexities. Run seamlessly on multi-core CPUs via Rayon, dedicated GPUs via Vulkan/Metal/DX12, or directly in the browser using WebAssembly and WebGPU.
* **Advanced LLM Meta:** Native implementation of PagedAttention for zero-fragmentation KV-cache memory management, enabling high-throughput inference for Transformer-based architectures.
* **Production Training Suite:** Includes a sophisticated Optim module featuring AdamW, learning rate schedulers, and a dynamic GradScaler for safe Automatic Mixed Precision training.
* **Distributed Native:** First-class support for Multi-GPU training topologies using Ring-AllReduce collective communication paradigms.

## Why use Rust for AI? 🦀

Rust's AI ecosystem is young, but it is real and growing quickly. Machine Learning is a special form of software where you need very high level abstractions as well as extremely fast execution time. Rust is the perfect candidate for this since it provides zero-cost abstractions to easily create neural network modules, and fine-grained control over memory to optimize every detail.Rust is versatile enough to tackle two-language dichotomy, and Cargo makes it easy to build, test and deploy from any environment, which is usually a pain in Other Language.

## Ecosystem Interoperability

Clove is designed to integrate into existing ML infrastructure.
* **C-ABI / FFI:** Exposes a safe C Application Binary Interface, allowing Clove to be driven as a high-performance backend for other Languages.
* **ONNX Export:** Built-in Protobuf visitor ONNXExporter allows any Clove computation graph to be instantly exported to .onnx for deployment to TensorRT or CoreML.

## Prerequisites

* **Rust:** Latest stable toolchain (install via rustup).
* **Hardware:** Vulkan, Metal, or DX12 compliant drivers for GPU acceleration.

## Contributing

We welcome community contributions. Please ensure that all new operations include corresponding WGSL shader implementations in backend.rs and appropriate test coverage.

## Status

Clove is currently in active development, and there will be breaking changes. While any resulting issues are likely to be easy to fix, there are no guarantees at this stage.

## License

Clove is distributed under the terms of the MIT license.
