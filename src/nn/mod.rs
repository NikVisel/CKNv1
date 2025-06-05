//! Neural network integrations and utilities

mod hopf_loss;
mod datasets;
mod edge_classifier;
mod tree_autoencoder;
mod tree_rl;
mod hopf_former;

pub use hopf_loss::HopfRegularizer;
pub use datasets::{AdmissibleCutsDataset, AntipodeDataset, GraftingDataset, CoefficientDataset};
pub use edge_classifier::{EdgeClassifier, EdgeAttention, EdgeClass, ClassificationMode};
pub use tree_autoencoder::{GraftedTreeAutoencoder, MaskedTreeReconstruction, AutoencoderConfig};
pub use tree_rl::{TreeState, TreeAction, TreeEnvironment, TreePolicy, Episode, generate_episode};
pub use hopf_former::{HopfFormer, HopfFormerConfig};
