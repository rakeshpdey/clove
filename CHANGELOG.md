Changelog

All notable changes to the Clove framework will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

[0.1.8] - 2026-07-10

Fixed

Shape Mismatch and RoPE Shader.

Update

Lazy.rs

[0.1.7] - 2026-07-07

Fixed

update matmul,transpose,cross-entropy,rope,softmax

[0.1.6] - 2026-07-06

Fixed

Resolved dropout implementation.

[0.1.4] - 2026-07-02

Added

Implemented core matrix multiplication optimization for matmul routine.
Added comprehensive unit tests for edge-case dimensions in linear algebra module.

Fixed

Resolved segmentation fault in memory-aligned buffer allocation.
Corrected off-by-one error in tiling cache logic.



[0.1.3] - 2026-07-01

Added

Universal C-ABI Gateway (ffi.rs): Exposed high-level modules including LanguageModel, DataLoader, GradScaler, and AdamW for Python/Julia/Go interoperability.

JIT Compiler Optimizations: Implemented true Algebraic Constant Folding and injected the TopK WGSL compute shader for Mixture-of-Experts routing.

Dynamic Memory Mapping: PagedAttention now dynamically parses head_dim directly from the tensor shape at runtime.

Fixed

FFI Routing: Bypassed TensorGraph wrapper limitations by mapping FFI math operations directly to <WgpuBackend>.

Thread Safety: Implemented Drop trait on DataLoader to safely join asynchronous background pre-fetching threads and prevent memory leaks.

[0.1.2] - 2026-06-30

Added

Universal FFI Bridge:

Exposed complete math suite and neural network primitives to foreign languages.
Implemented `clove_optimizer_create`, `step`, and `zero_grad` for full GPU-side training loops.
Added strict `# Safety` documentation for all FFI operations to meet Rust production standards.

[0.1.1] - 2026-06-29

Added

Performance:

Optimized `rayon_matmul` via transpose-based cache locality, achieving 13x speedup on 1024x1024 matrices.

[0.1.0] - 2026-06-29

Added

Core Engine:

Initial release of the Clove Engine.

LazyBackend JIT compiler with WGSL kernel fusion.

Full Autograd calculus engine supporting higher-order derivatives.

Core Neural Network (nn) modules including Linear, TransformerBlock, MoELayer, and PagedAttention.

Phase 1-5 Math operators (Arithmetic, Trigonometry, Pooling, Conv1d/2d/3d, Geometry).

C ABI Bridge (ffi.rs) for Python/C++/Go interoperability.

Asynchronous Ring-Buffer DataLoader.

High-Performance Infrastructure:

Multi-GPU Ring-AllReduce distributed training topology.

Advanced memory pooling for WebGPU buffers to eliminate driver allocation overhead.

Resolved VRAM memory thrashing via optimized buffer reuse.

Corrected initial layer normalization weight scaling.