//! Core functionality bridging Hopf algebra and machine learning

mod hopf_ml;
mod model_config;
mod training;
mod hopf_invariant;
mod neural_hopf_flows;
mod geometric_embeddings;

pub use hopf_ml::HopfML;
pub use model_config::{ModelConfig, TaskType, LossFunction};
pub use training::{TrainingConfig, Trainer, TrainingMetrics};
pub use hopf_invariant::{HopfInvariantLoss, AlgebraicConstraints};
pub use neural_hopf_flows::{HopfFlow, NeuralODE, GeometricHopfFlow};
pub use geometric_embeddings::{CGA, GeometricEmbedding, HyperbolicEmbedding, SphericalEmbedding};
