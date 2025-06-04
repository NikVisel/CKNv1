//! Training infrastructure for Hopf-ML models

use crate::graph::GraphData;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Training configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Train/validation split ratio
    pub validation_split: f32,
    /// Whether to shuffle data
    pub shuffle: bool,
    /// Device to use (cpu/cuda)
    pub device: String,
    /// Checkpoint frequency (epochs)
    pub checkpoint_every: Option<usize>,
    /// Verbose logging
    pub verbose: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            seed: Some(42),
            validation_split: 0.2,
            shuffle: true,
            device: "cpu".to_string(),
            checkpoint_every: Some(10),
            verbose: true,
        }
    }
}

/// Training metrics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss history
    pub train_loss: Vec<f32>,
    /// Validation loss history
    pub val_loss: Vec<f32>,
    /// Additional metrics (e.g., accuracy, MAE)
    pub custom_metrics: std::collections::HashMap<String, Vec<f32>>,
    /// Best validation loss
    pub best_val_loss: f32,
    /// Epoch with best validation loss
    pub best_epoch: usize,
    /// Total training time
    pub total_time: Duration,
}

impl TrainingMetrics {
    /// Create new metrics tracker
    pub fn new() -> Self {
        TrainingMetrics {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            custom_metrics: std::collections::HashMap::new(),
            best_val_loss: f32::INFINITY,
            best_epoch: 0,
            total_time: Duration::from_secs(0),
        }
    }

    /// Update metrics for an epoch
    pub fn update_epoch(&mut self, epoch: usize, train_loss: f32, val_loss: f32) {
        self.train_loss.push(train_loss);
        self.val_loss.push(val_loss);
        
        if val_loss < self.best_val_loss {
            self.best_val_loss = val_loss;
            self.best_epoch = epoch;
        }
    }

    /// Add custom metric
    pub fn add_metric(&mut self, name: &str, value: f32) {
        self.custom_metrics
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(value);
    }

    /// Check if should stop early
    pub fn should_stop_early(&self, patience: usize) -> bool {
        let current_epoch = self.train_loss.len();
        current_epoch > self.best_epoch + patience
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        format!(
            "Best validation loss: {:.4} at epoch {}\nTotal training time: {:.2}s\nFinal train loss: {:.4}\nFinal val loss: {:.4}",
            self.best_val_loss,
            self.best_epoch,
            self.total_time.as_secs_f32(),
            self.train_loss.last().unwrap_or(&0.0),
            self.val_loss.last().unwrap_or(&0.0)
        )
    }
}

/// Generic trainer interface
pub trait Trainer {
    /// Model type
    type Model;
    /// Input data type
    type Input;
    /// Target type
    type Target;
    
    /// Initialize model
    fn initialize_model(&self) -> Self::Model;
    
    /// Train for one epoch
    fn train_epoch(
        &mut self,
        model: &mut Self::Model,
        data: &[(Self::Input, Self::Target)],
    ) -> f32;
    
    /// Evaluate on validation data
    fn evaluate(
        &self,
        model: &Self::Model,
        data: &[(Self::Input, Self::Target)],
    ) -> f32;
    
    /// Save model checkpoint
    fn save_checkpoint(&self, model: &Self::Model, path: &str) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Load model checkpoint
    fn load_checkpoint(&self, path: &str) -> Result<Self::Model, Box<dyn std::error::Error>>;
}

/// Data loader for batching
pub struct DataLoader<I, T> {
    data: Vec<(I, T)>,
    batch_size: usize,
    shuffle: bool,
    current_idx: usize,
}

impl<I: Clone, T: Clone> DataLoader<I, T> {
    /// Create new data loader
    pub fn new(data: Vec<(I, T)>, batch_size: usize, shuffle: bool) -> Self {
        let mut loader = DataLoader {
            data,
            batch_size,
            shuffle,
            current_idx: 0,
        };
        
        if shuffle {
            loader.shuffle_data();
        }
        
        loader
    }

    /// Shuffle data
    fn shuffle_data(&mut self) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.data.shuffle(&mut rng);
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            self.shuffle_data();
        }
    }

    /// Get next batch
    pub fn next_batch(&mut self) -> Option<Vec<(I, T)>> {
        if self.current_idx >= self.data.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.data.len());
        let batch = self.data[self.current_idx..end_idx].to_vec();
        self.current_idx = end_idx;

        Some(batch)
    }

    /// Get number of batches
    pub fn num_batches(&self) -> usize {
        (self.data.len() + self.batch_size - 1) / self.batch_size
    }
}

/// Split data into train and validation sets
pub fn train_val_split<I: Clone, T: Clone>(
    data: Vec<(I, T)>,
    val_ratio: f32,
    shuffle: bool,
) -> (Vec<(I, T)>, Vec<(I, T)>) {
    let mut data = data;
    
    if shuffle {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        data.shuffle(&mut rng);
    }
    
    let val_size = (data.len() as f32 * val_ratio) as usize;
    let train_size = data.len() - val_size;
    
    let train_data = data[..train_size].to_vec();
    let val_data = data[train_size..].to_vec();
    
    (train_data, val_data)
}

/// Generic training loop
pub fn train_loop<TR, I, T>(
    mut trainer: TR,
    data: Vec<(I, T)>,
    config: TrainingConfig,
    model_config: &crate::core::ModelConfig,
) -> (TR::Model, TrainingMetrics)
where
    TR: Trainer<Input = I, Target = T>,
    I: Clone,
    T: Clone,
{
    // Set random seed if provided
    if let Some(seed) = config.seed {
        use rand::SeedableRng;
        let _rng = rand::rngs::StdRng::seed_from_u64(seed);
    }
    
    // Split data
    let (train_data, val_data) = train_val_split(data, config.validation_split, config.shuffle);
    
    if config.verbose {
        println!("Training set size: {}", train_data.len());
        println!("Validation set size: {}", val_data.len());
    }
    
    // Initialize model and metrics
    let mut model = trainer.initialize_model();
    let mut metrics = TrainingMetrics::new();
    let start_time = Instant::now();
    
    // Training loop
    for epoch in 0..model_config.num_epochs {
        // Train epoch
        let mut train_loader = DataLoader::new(
            train_data.clone(),
            model_config.batch_size,
            config.shuffle,
        );
        
        let mut epoch_loss = 0.0;
        let num_batches = train_loader.num_batches();
        
        while let Some(batch) = train_loader.next_batch() {
            let batch_loss = trainer.train_epoch(&mut model, &batch);
            epoch_loss += batch_loss;
        }
        
        let avg_train_loss = epoch_loss / num_batches as f32;
        
        // Validation
        let val_loss = trainer.evaluate(&model, &val_data);
        
        // Update metrics
        metrics.update_epoch(epoch, avg_train_loss, val_loss);
        
        if config.verbose && epoch % 10 == 0 {
            println!(
                "Epoch {}/{}: train_loss={:.4}, val_loss={:.4}",
                epoch + 1,
                model_config.num_epochs,
                avg_train_loss,
                val_loss
            );
        }
        
        // Checkpointing
        if let Some(freq) = config.checkpoint_every {
            if (epoch + 1) % freq == 0 {
                let path = format!("checkpoint_epoch_{}.pt", epoch + 1);
                if let Err(e) = trainer.save_checkpoint(&model, &path) {
                    eprintln!("Failed to save checkpoint: {}", e);
                }
            }
        }
        
        // Early stopping
        if let Some(patience) = model_config.early_stopping_patience {
            if metrics.should_stop_early(patience) {
                if config.verbose {
                    println!("Early stopping at epoch {}", epoch + 1);
                }
                break;
            }
        }
    }
    
    metrics.total_time = start_time.elapsed();
    
    if config.verbose {
        println!("\nTraining complete!");
        println!("{}", metrics.summary());
    }
    
    (model, metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_loader() {
        let data: Vec<(i32, i32)> = vec![(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)];
        let mut loader = DataLoader::new(data, 2, false);
        
        assert_eq!(loader.num_batches(), 3);
        
        let batch1 = loader.next_batch().unwrap();
        assert_eq!(batch1.len(), 2);
        assert_eq!(batch1[0], (1, 2));
        
        let batch2 = loader.next_batch().unwrap();
        assert_eq!(batch2.len(), 2);
        
        let batch3 = loader.next_batch().unwrap();
        assert_eq!(batch3.len(), 1);
        assert_eq!(batch3[0], (9, 10));
        
        assert!(loader.next_batch().is_none());
    }

    #[test]
    fn test_metrics() {
        let mut metrics = TrainingMetrics::new();
        
        metrics.update_epoch(0, 1.0, 0.9);
        metrics.update_epoch(1, 0.8, 0.7);
        metrics.update_epoch(2, 0.6, 0.8);
        
        assert_eq!(metrics.best_epoch, 1);
        assert_eq!(metrics.best_val_loss, 0.7);
        assert!(metrics.should_stop_early(1)); // patience=1, current=2, best=1
    }
} 