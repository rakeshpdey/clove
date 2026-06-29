Changelog

All notable changes to the Clove framework will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

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