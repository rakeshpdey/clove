Clove

Clove is a high-performance machine learning and deep learning framework. Designed for both scale-out cluster training and edge-native inference, it leverages hardware acceleration via wgpu, lazy evaluation for kernel fusion, and a robust tape-based autograd engine.

Quick Links

* [**View Source on GitHub**](https://github.com/rakeshpdey/clove)
* [**Report Issues / Request Features**](https://github.com/rakeshpdey/clove/issues)
* [**API Documentation**](https://docs.rs/clove)

Quick Start

use clove::nn::Linear;
use clove::tensor::Tensor;
use ndarray::array;

// Build a layer and a tensor instantly
let layer = Linear::new(2, 1);
let input = Tensor::new(array![[1.0, 2.0]]);
let prediction = layer.forward(&input);


Core Architecture

Lazy Execution & JIT Compilation: Clove dynamically traces computation graphs into an Intermediate Representation. The LazyEngine performs dead-code elimination, constant folding, and horizontal kernel fusion before compiling the graph into hyper-optimized WGSL shaders.

Hardware Agnostic (CPU/GPU/WASM): A unified Backend trait abstracts hardware complexities. Run seamlessly on multi-core CPUs via Rayon, dedicated GPUs via Vulkan/Metal/DX12, or directly in the browser using WebAssembly and WebGPU.

Advanced LLM Meta: Native implementation of PagedAttention for zero-fragmentation KV-cache memory management, enabling high-throughput inference for Transformer-based architectures.

Production Training Suite: Includes a sophisticated Optim module featuring AdamW, learning rate schedulers, and a dynamic GradScaler for safe Automatic Mixed Precision training.

Distributed Native: First-class support for Multi-GPU training topologies using Ring-AllReduce collective communication paradigms.

Ecosystem Interoperability

Clove is designed to integrate into existing ML infrastructure.

C-ABI / FFI: Exposes a safe C Application Binary Interface, allowing Clove to be driven as a high-performance backend for other Languages.

ONNX Export: Built-in Protobuf visitor ONNXExporter allows any Clove computation graph to be instantly exported to .onnx for deployment to TensorRT or CoreML.

Prerequisites

Rust: Latest stable toolchain (install via rustup).

Hardware: Vulkan, Metal, or DX12 compliant drivers for GPU acceleration.

Contributing

We welcome community contributions. Please ensure that all new operations include corresponding WGSL shader implementations in backend.rs and appropriate test coverage.

Status

Clove is currently in active development, and there will be breaking changes. While any resulting issues are likely to be easy to fix, there are no guarantees at this stage.

License

Clove is distributed under the terms of the MIT license.