//! Model configuration for Hopf-ML tasks

use serde::{Serialize, Deserialize};

/// Type of ML task
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
    /// Regression (e.g., predicting number of cuts)
    Regression,
    /// Classification (e.g., edge cut prediction)
    Classification { num_classes: usize },
    /// Contrastive learning (e.g., grafting similarity)
    Contrastive,
    /// Multi-task learning
    MultiTask(Vec<TaskType>),
}

/// Loss function type
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LossFunction {
    /// Mean squared error
    MSE,
    /// Mean absolute error
    MAE,
    /// Cross entropy
    CrossEntropy,
    /// Binary cross entropy
    BCE,
    /// Contrastive loss
    Contrastive { margin: f32 },
    /// Custom weighted combination
    Weighted(Vec<(LossFunction, f32)>),
}

/// Neural network architecture type
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Architecture {
    /// Graph Neural Network
    GNN {
        num_layers: usize,
        hidden_dim: usize,
        aggregation: AggregationType,
    },
    /// Tree Transformer
    Transformer {
        num_layers: usize,
        num_heads: usize,
        hidden_dim: usize,
        dropout: f32,
    },
    /// Hybrid GNN + Transformer
    Hybrid {
        gnn_layers: usize,
        transformer_layers: usize,
        hidden_dim: usize,
    },
}

/// Aggregation type for GNN
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AggregationType {
    Mean,
    Sum,
    Max,
    Attention,
}

/// Model configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Task type
    pub task: TaskType,
    /// Loss function
    pub loss: LossFunction,
    /// Architecture
    pub architecture: Architecture,
    /// Input feature dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub num_epochs: usize,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
    /// Use Hopf-algebraic regularization
    pub hopf_regularization: bool,
    /// Regularization weight
    pub reg_weight: f32,
}

impl ModelConfig {
    /// Create config for cuts prediction task
    pub fn for_cuts_prediction(max_tree_size: usize) -> Self {
        ModelConfig {
            task: TaskType::Regression,
            loss: LossFunction::MSE,
            architecture: Architecture::GNN {
                num_layers: 3,
                hidden_dim: 64,
                aggregation: AggregationType::Mean,
            },
            input_dim: 6, // Standard node features
            output_dim: 1,
            learning_rate: 0.001,
            batch_size: 32,
            num_epochs: 200,
            early_stopping_patience: Some(20),
            hopf_regularization: false,
            reg_weight: 0.01,
        }
    }

    /// Create config for antipode prediction
    pub fn for_antipode_prediction(max_tree_size: usize) -> Self {
        ModelConfig {
            task: TaskType::Regression,
            loss: LossFunction::MSE,
            architecture: Architecture::Transformer {
                num_layers: 4,
                num_heads: 4,
                hidden_dim: 128,
                dropout: 0.1,
            },
            input_dim: 8, // With Hopf features
            output_dim: max_tree_size + 1,
            learning_rate: 0.0005,
            batch_size: 16,
            num_epochs: 300,
            early_stopping_patience: Some(30),
            hopf_regularization: true,
            reg_weight: 0.1,
        }
    }

    /// Create config for contrastive grafting
    pub fn for_grafting_contrastive() -> Self {
        ModelConfig {
            task: TaskType::Contrastive,
            loss: LossFunction::Contrastive { margin: 1.0 },
            architecture: Architecture::Hybrid {
                gnn_layers: 2,
                transformer_layers: 2,
                hidden_dim: 96,
            },
            input_dim: 6,
            output_dim: 96, // Embedding dimension
            learning_rate: 0.0001,
            batch_size: 64,
            num_epochs: 100,
            early_stopping_patience: Some(15),
            hopf_regularization: true,
            reg_weight: 0.05,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.input_dim == 0 {
            return Err("Input dimension must be positive".to_string());
        }
        
        if self.output_dim == 0 {
            return Err("Output dimension must be positive".to_string());
        }
        
        if self.learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        
        if self.batch_size == 0 {
            return Err("Batch size must be positive".to_string());
        }
        
        match &self.architecture {
            Architecture::GNN { num_layers, hidden_dim, .. } => {
                if *num_layers == 0 {
                    return Err("GNN must have at least one layer".to_string());
                }
                if *hidden_dim == 0 {
                    return Err("Hidden dimension must be positive".to_string());
                }
            }
            Architecture::Transformer { num_layers, num_heads, hidden_dim, .. } => {
                if *num_layers == 0 {
                    return Err("Transformer must have at least one layer".to_string());
                }
                if *num_heads == 0 {
                    return Err("Number of heads must be positive".to_string());
                }
                if hidden_dim % num_heads != 0 {
                    return Err("Hidden dimension must be divisible by number of heads".to_string());
                }
            }
            Architecture::Hybrid { gnn_layers, transformer_layers, hidden_dim } => {
                if *gnn_layers == 0 && *transformer_layers == 0 {
                    return Err("Hybrid model must have at least one layer".to_string());
                }
                if *hidden_dim == 0 {
                    return Err("Hidden dimension must be positive".to_string());
                }
            }
        }
        
        Ok(())
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::for_cuts_prediction(6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let mut config = ModelConfig::default();
        assert!(config.validate().is_ok());
        
        config.input_dim = 0;
        assert!(config.validate().is_err());
        
        config.input_dim = 6;
        config.learning_rate = -0.001;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_predefined_configs() {
        let cuts_config = ModelConfig::for_cuts_prediction(10);
        assert!(cuts_config.validate().is_ok());
        assert_eq!(cuts_config.output_dim, 1);
        
        let antipode_config = ModelConfig::for_antipode_prediction(5);
        assert!(antipode_config.validate().is_ok());
        assert_eq!(antipode_config.output_dim, 6); // 0..5 inclusive
        
        let contrastive_config = ModelConfig::for_grafting_contrastive();
        assert!(contrastive_config.validate().is_ok());
    }
} 