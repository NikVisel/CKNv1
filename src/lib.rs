//! # Hopf-ML: Hopf Algebras meet Machine Learning
//! 
//! This library implements the mathematical structures from Connes-Kreimer's
//! Hopf algebra of rooted trees, with neural network integration for learning
//! algebraic patterns and renormalization structures.
//!
//! ## Features
//! 
//! - **Core Hopf Algebra**: Rooted trees, coproduct, antipode operations
//! - **Neural Networks**: GNN and Transformer architectures for trees
//! - **Applications**: Renormalization, graph generation, algebraic learning
//! - **Visualization**: Interactive tree and algebra visualization
//! - **Cross-platform**: Works on CPU, GPU (with features), and WASM

#![warn(missing_docs)]
#![warn(clippy::all)]

/// Core algebraic structures and operations
pub mod algebra;

/// Tree and graph data structures
pub mod graph;

/// Neural network architectures and training
pub mod nn;

/// Core functionality that bridges algebra and ML
pub mod core;

/// Utility functions and helpers
pub mod utils;

/// Application-specific implementations
pub mod applications;

// Re-export commonly used types
pub use algebra::{HopfAlgebra, Tree, Forest, CoProduct, Antipode};
pub use graph::{GraphData, TreeGraph, tree_to_graph};
pub use core::{HopfML, ModelConfig};

#[cfg(feature = "nn-tch")]
pub use nn::tch_impl::{TreeGNN, TreeTransformer};

/// Error types for the library
#[derive(Debug, thiserror::Error)]
pub enum HopfMLError {
    /// Invalid tree structure
    #[error("Invalid tree: {0}")]
    InvalidTree(String),
    
    /// Algebraic operation error
    #[error("Algebra error: {0}")]
    AlgebraError(String),
    
    /// Neural network error
    #[error("Neural network error: {0}")]
    NeuralNetError(String),
    
    /// IO or serialization error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for the library
pub type Result<T> = std::result::Result<T, HopfMLError>;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        algebra::{Tree, Forest, HopfAlgebra, CoProduct, Antipode, TreeBuilder},
        graph::{GraphData, TreeGraph, tree_to_graph},
        core::{HopfML, ModelConfig},
        Result, HopfMLError,
    };
    
    #[cfg(feature = "nn-tch")]
    pub use crate::nn::tch_impl::{TreeGNN, TreeTransformer};
}
