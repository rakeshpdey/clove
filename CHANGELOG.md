Changelog

All notable changes to the Clove framework will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

[Unreleased]

Added

Multi-GPU Ring-AllReduce distributed training topology.

Advanced memory pooling for WebGPU buffers to eliminate driver allocation overhead.

[1.0.0] - 2026-06-29

Added

Initial release of the Clove Engine.

LazyBackend JIT compiler with WGSL kernel fusion.

Full Autograd calculus engine supporting higher-order derivatives.

Core Neural Network (nn) modules including Linear, TransformerBlock, MoELayer, and PagedAttention.

Phase 1-5 Math operators (Arithmetic, Trigonometry, Pooling, Conv1d/2d/3d, Geometry).

C ABI Bridge (ffi.rs) for Python/C++/GO/ interoperability.

Asynchronous Ring-Buffer DataLoader.