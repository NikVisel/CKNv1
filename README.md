# Hopf-ML: Hopf Algebras meet Machine Learning

A Rust library implementing the mathematical structures from Connes-Kreimer's Hopf algebra of rooted trees, with neural network integration for learning algebraic patterns and renormalization structures.

## Overview

This library brings together:
- **Pure Mathematics**: Complete implementation of the Hopf algebra of rooted trees
- **Machine Learning**: GNN and Transformer architectures for learning on trees
- **Physics Applications**: Renormalization group flows and BPHZ forest formulas
- **High Performance**: Parallel algorithms, caching, and optional GPU support

## Features

### Core Algebra
- Rooted tree data structures with efficient operations
- Coproduct computation via admissible cuts
- Antipode calculation with memoization
- Generation of δ_k (sum of all trees with k vertices)
- Forest operations and Hopf algebra elements

### Machine Learning Integration
- Convert trees to graph data for neural networks
- Rich node features (structural, topological, Hopf-algebraic)
- Support for GNN, Transformer, and hybrid architectures
- Training infrastructure with early stopping and checkpointing
- Pre-built tasks: cuts prediction, antipode learning, contrastive grafting

### Applications
- Renormalization simulation with forest formulas
- Tree generation via iterative grafting
- Algebraic pattern discovery
- Dataset generation for various ML tasks

## Quick Start

```rust
use hopf_ml::prelude::*;

// Create a rooted tree
let tree = TreeBuilder::new()
    .add_child(0, 1)
    .add_child(0, 2)
    .add_child(1, 3)
    .build()
    .unwrap();

// Compute Hopf operations
let coproduct = tree.coproduct();
let antipode = tree.antipode();

// Convert to ML-ready format
let graph_data = tree_to_graph(&tree);

// Use with HopfML
let mut hopf_ml = HopfML::new();
let dataset = hopf_ml.generate_cuts_dataset(6);
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
hopf-ml = "0.1"

# Optional features
hopf-ml = { version = "0.1", features = ["nn-tch", "viz"] }
```

### Features
- `nn-tch`: PyTorch backend via tch-rs
- `nn-burn`: Pure Rust neural networks via Burn
- `nn-candle`: Candle backend
- `viz`: Visualization support
- `wasm`: WebAssembly support

## Examples

### Basic Hopf Algebra Operations

```rust
// Generate all trees of size 4
let trees = delta_k(4);
println!("There are {} rooted trees with 4 nodes", trees.len());

// Compute admissible cuts
for tree in &trees {
    let cuts = tree.admissible_cuts();
    println!("Tree has {} admissible cuts", cuts.len());
}
```

### Machine Learning Pipeline

```rust
// Configure model for predicting number of cuts
let config = ModelConfig::for_cuts_prediction(8);

// Generate dataset
let mut hopf_ml = HopfML::new();
let data = hopf_ml.generate_cuts_dataset(8);

// Train model (with appropriate backend)
#[cfg(feature = "nn-tch")]
{
    let trainer = CutsTrainer::new(config);
    let (model, metrics) = train_loop(trainer, data, Default::default());
    println!("Best validation loss: {}", metrics.best_val_loss);
}
```

### Advanced: Hopf-Algebraic Regularization

```rust
// Create a model with Hopf-algebraic constraints
let config = ModelConfig::for_antipode_prediction(6)
    .with_hopf_regularization(true);

// The model will learn while respecting:
// - Antipode involution: S(S(t)) = t
// - Coproduct consistency
// - Natural growth equivariance
```

## Architecture

```
hopf-ml/
├── src/
│   ├── algebra/        # Core Hopf algebra implementation
│   ├── graph/          # Tree-to-graph conversion
│   ├── nn/             # Neural network modules
│   ├── core/           # Integration layer
│   ├── applications/   # Example applications
│   └── utils/          # Helper functions
├── examples/           # Runnable examples
├── benches/           # Performance benchmarks
└── tests/             # Integration tests
```

## Mathematical Background

This library implements the Hopf algebra H_R from Connes-Kreimer (1998):

- **Trees**: Rooted trees as algebraic generators
- **Coproduct**: Δ(t) = t⊗1 + 1⊗t + Σ P_C(t)⊗R_C(t) over admissible cuts
- **Antipode**: S(t) = -t - Σ S(P_C(t))·R_C(t)
- **Natural Growth**: The operator N that attaches leaves

The library makes these abstract concepts concrete and computable.

## Use Cases

1. **Research in Mathematical Physics**
   - Study renormalization group flows
   - Compute BPHZ counterterms
   - Explore Hopf algebra structures

2. **Machine Learning Research**
   - Graph neural networks on algebraic structures
   - Learning combinatorial patterns
   - Invariant and equivariant architectures

3. **Educational Tool**
   - Visualize Hopf operations
   - Experiment with tree algebras
   - Bridge pure math and ML

4. **Applications**
   - Quantum field theory calculations
   - Combinatorial optimization
   - Symbolic computation

## Performance

- Parallel computation of admissible cuts for large trees
- Memoized antipode calculations
- Efficient tree isomorphism checking
- GPU acceleration available for neural networks

## Contributing

Contributions are welcome! Areas of interest:
- Additional neural architectures
- More physics applications
- Visualization improvements
- Performance optimizations
- Documentation and tutorials

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## References

- Connes, A., & Kreimer, D. (1998). "Hopf algebras, renormalization and noncommutative geometry"
- Additional papers and resources in the [docs/](docs/) directory

## Acknowledgments

This implementation is inspired by the beautiful mathematics of Alain Connes and Dirk Kreimer, bridging abstract algebra with concrete computational applications. 
For details on dataset preparation and training see [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).
