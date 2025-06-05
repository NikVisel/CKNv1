# Hopf-ML Production Architecture

This document sketches an opinionated production architecture for the Hopf-ML project. The design highlights several existing components from the codebase that provide a solid foundation for a scalable system combining Hopf algebra computations with modern machine learning techniques.

## 1. Core Library (`hopf-ml` crate)

The core crate exposes the fundamental algebraic structures together with utilities for machine learning integration.

- **Algebra module** – Implements rooted trees, forests, the coproduct and antipode, and supports natural growth through `delta_k` generation.
- **Graph module** – Converts trees to graph representations suitable for GNNs or transformers.
- **Core module** – Provides training infrastructure, model configuration and the `HopfInvariantLoss` for embedding regularization.
- **Applications module** – Contains higher level examples such as renormalization, quantum simulations and the dual-stream transformer architecture.

These modules are designed for high performance and can optionally enable GPU acceleration through feature flags (`nn-tch`, `nn-candle`, etc.).

## 2. Embedding and Model Pipeline

A central idea is to fuse algebraic and geometric views of the trees. The `DualStreamTransformer` from `applications/dual_stream.rs` demonstrates this approach by running two attention streams in parallel—one for algebraic features and one for geometric embeddings—and then combining them.

Training is guided by the `HopfInvariantLoss` which enforces product linearity, antipode equivariance and coproduct consistency. Together these components form a powerful pipeline for learning representations that respect Hopf structure while remaining expressive for downstream tasks.

```
Tree data ─┐             ┌─> Algebra stream (structural features)
           ├─> DualStreamTransformer ──> Combined embeddings
Tree data ─┘             └─> Geometry stream (geometric embeddings)
```

The pipeline can be driven by application-specific trainers, e.g. `DualStreamTrainer`, which wrap the transformer and loss computation. Datasets may be generated on the fly using the `graph_generation` and `algebraic_learning` modules.

## 3. Modular Applications

Each submodule under `src/applications` targets a different domain:

- **Renormalization** – BPHZ forest formulas and flow simulations.
- **Quantum** – Quantum circuit models leveraging tree coproducts.
- **Molecular** – Exploratory support for chemical graph learning.
- **Blockchain** – Algebraic validation of block histories.

These examples demonstrate how the core algebra and learning components can be reused in diverse settings. In production we can expose them as separate binaries or library entry points, sharing a common set of embeddings and training utilities.

## 4. Suggested Architecture Layers

1. **Data & Algebra Layer**
   - Responsible for efficient Hopf-algebraic operations (`Tree`, `Forest`, coproduct, antipode) and tree generation.
   - Provides dataset generation utilities and graph conversions.

2. **Model Layer**
   - Implements embedding networks (GNNs, dual-stream transformer, neural Hopf flows).
   - Houses the `HopfInvariantLoss` and regularization techniques.

3. **Application Layer**
   - Domain-specific services or binaries (renormalization, quantum, etc.).
   - Reuse the model layer to provide specialized training loops and inference.

4. **Interface Layer**
   - Bindings (e.g. via `pyo3` or `wasm`) to expose selected functionality to Python or the Web.
   - CLI tools for dataset generation and model training.

5. **Infrastructure Layer**
   - Continuous integration to build and test the project (`cargo check`, `cargo test` workflows).
   - Optional containerization for deployment, with GPU support when available.

## 5. HopfFormer Architecture

Version 0.1.0 introduces **HopfFormer**, a multi-layer transformer that stacks
the existing dual-stream blocks. Each layer runs algebraic and geometric
attention in parallel and mixes the streams with cross-attention. HopfFormer
exposes a simple `HopfFormerConfig` for choosing the embedding dimension,
number of layers and attention heads. The model interfaces with
`HopfInvariantLoss` to enforce algebraic consistency during training.

```
trees ──> AlgebraStream ──┐
                          │                 ┌─> layer 1
trees ──> GeometryStream ─┼─> HopfFormer ───┼─> layer 2
                          │                 └─> ...
                          ▼
                Combined embeddings
```

HopfFormer serves as the recommended baseline going forward.

## 6. Next Steps

The project now compiles and all unit tests pass. Remaining work focuses on polishing the API and expanding documentation:

- Finalize the dual-stream transformer API and training loops.
- Provide preconfigured datasets and example experiments in `examples/`.
- Offer Python bindings for easier experimentation.
- Document best practices for extending the Hopf algebra types and neural modules.

By iteratively integrating these components we move toward a production-ready system that marries rich algebraic structure with modern machine learning.

