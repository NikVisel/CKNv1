//! Applications of Hopf algebra in machine learning

mod renormalization;
mod graph_generation;
mod algebraic_learning;
mod dual_stream;
mod blockchain;
mod quantum;
mod molecular;

pub use renormalization::RenormalizationSimulator;
pub use graph_generation::HopfTreeGenerator;
pub use algebraic_learning::AlgebraicLearner;
pub use dual_stream::{DualStreamTransformer, AttentionMechanism, AlgebraStream, GeometryStream};
pub use blockchain::{HopfChain, HopfContract, AlgebraicTransaction, HopfOperation, AlgebraicObject, ComputationProof, Block};
pub use quantum::{QuantumTreeState, HopfGate, HopfQuantumCircuit, HopfVQE, TreeObservable}; 