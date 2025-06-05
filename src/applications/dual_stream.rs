//! Dual-stream transformer architecture with algebra and geometry streams

use crate::algebra::{Tree, Forest, CoProduct, Antipode};
use crate::core::{HopfInvariantLoss, GeometricEmbedding, AlgebraicConstraints};
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

/// Attention mechanism types
#[derive(Debug, Clone)]
pub enum AttentionMechanism {
    /// Standard scaled dot-product attention
    Standard,
    /// Hyperbolic attention using Poincaré distance
    Hyperbolic,
    /// Attention masked by tree structure
    TreeMasked,
    /// Attention weighted by algebraic operations
    AlgebraWeighted,
}

/// Configuration for [`DualStreamTransformer`]
#[derive(Debug, Clone)]
pub struct DualStreamConfig {
    /// Dimension of embeddings
    pub embed_dim: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Attention mechanism for the algebra stream
    pub algebra_attention: AttentionMechanism,
    /// Attention mechanism for the geometry stream
    pub geometry_attention: AttentionMechanism,
}

impl Default for DualStreamConfig {
    fn default() -> Self {
        DualStreamConfig {
            embed_dim: 8,
            n_heads: 1,
            algebra_attention: AttentionMechanism::Standard,
            geometry_attention: AttentionMechanism::Standard,
        }
    }
}

/// Dual-stream transformer combining algebraic and geometric processing
pub struct DualStreamTransformer {
    /// Dimension of embeddings
    embed_dim: usize,
    /// Number of attention heads
    n_heads: usize,
    /// Attention mechanism for each stream
    algebra_attention: AttentionMechanism,
    geometry_attention: AttentionMechanism,
    /// Cross-stream interaction weights
    cross_weights: Array2<f32>,
}

impl DualStreamTransformer {
    /// Create new dual-stream transformer from [`DualStreamConfig`]
    pub fn new(cfg: DualStreamConfig) -> Self {
        let DualStreamConfig {
            embed_dim,
            n_heads,
            algebra_attention,
            geometry_attention,
        } = cfg;
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let cross_weights = Array2::from_shape_fn((embed_dim, embed_dim), |_| {
            rng.gen_range(-0.1..0.1) / (embed_dim as f32).sqrt()
        });
        
        DualStreamTransformer {
            embed_dim,
            n_heads,
            algebra_attention,
            geometry_attention,
            cross_weights,
        }
    }

    /// Forward pass through dual streams
    pub fn forward(
        &self,
        trees: &[Tree],
        algebra_embeds: Array2<f32>,
        geometry_embeds: Array2<f32>,
    ) -> DualStreamOutput {
        // Process through algebra stream
        let algebra_out = self.algebra_stream(trees, &algebra_embeds);
        
        // Process through geometry stream
        let geometry_out = self.geometry_stream(trees, &geometry_embeds);
        
        // Cross-stream interaction
        let cross_algebra = self.cross_attention(&algebra_out, &geometry_out);
        let cross_geometry = self.cross_attention(&geometry_out, &algebra_out);
        
        // Combine streams
        let combined = self.combine_streams(&cross_algebra, &cross_geometry);
        
        DualStreamOutput {
            algebra_features: cross_algebra,
            geometry_features: cross_geometry,
            combined_features: combined,
            attention_weights: self.compute_attention_weights(trees),
        }
    }

    /// Convenience forward pass that internally computes embeddings
    pub fn forward_with_streams(
        &self,
        trees: &[Tree],
        algebra_stream: &AlgebraStream,
        geom_stream: &GeometryStream,
    ) -> DualStreamOutput {
        let mut algebra_embeds = Array2::zeros((trees.len(), self.embed_dim));
        let mut geometry_embeds = Array2::zeros((trees.len(), self.embed_dim));

        for (i, tree) in trees.iter().enumerate() {
            algebra_embeds.row_mut(i).assign(&algebra_stream.process(tree));
            geometry_embeds.row_mut(i).assign(&geom_stream.embed(tree));
        }

        self.forward(trees, algebra_embeds, geometry_embeds)
    }

    fn algebra_stream(&self, trees: &[Tree], embeds: &Array2<f32>) -> Array2<f32> {
        match &self.algebra_attention {
            AttentionMechanism::Standard => self.standard_attention(embeds),
            AttentionMechanism::TreeMasked => self.tree_masked_attention(trees, embeds),
            AttentionMechanism::AlgebraWeighted => self.algebra_weighted_attention(trees, embeds),
            _ => self.standard_attention(embeds),
        }
    }

    fn geometry_stream(&self, trees: &[Tree], embeds: &Array2<f32>) -> Array2<f32> {
        match &self.geometry_attention {
            AttentionMechanism::Standard => self.standard_attention(embeds),
            AttentionMechanism::Hyperbolic => self.hyperbolic_attention(embeds),
            _ => self.standard_attention(embeds),
        }
    }

    fn standard_attention(&self, embeds: &Array2<f32>) -> Array2<f32> {
        // Simplified implementation
        embeds.clone()
    }

    fn tree_masked_attention(&self, trees: &[Tree], embeds: &Array2<f32>) -> Array2<f32> {
        let mut out = embeds.clone();

        for (i, tree) in trees.iter().enumerate() {
            let leaf_ratio = tree.leaf_count() as f32 / tree.size().max(1) as f32;
            out.row_mut(i).mapv_inplace(|x| x * (1.0 - leaf_ratio));
        }

        out
    }

    fn algebra_weighted_attention(&self, trees: &[Tree], embeds: &Array2<f32>) -> Array2<f32> {
        let mut out = embeds.clone();

        for (i, tree) in trees.iter().enumerate() {
            let cop_len = tree.coproduct().len() as f32;
            let sign = if tree.antipode().trees().len() % 2 == 0 { 1.0 } else { -1.0 };
            out.row_mut(i).mapv_inplace(|x| x * sign * (cop_len + 1.0).ln());
        }

        out
    }

    fn hyperbolic_attention(&self, embeds: &Array2<f32>) -> Array2<f32> {
        // Map to Poincaré ball
        embeds.mapv(|x| x.tanh())
    }

    fn cross_attention(&self, query: &Array2<f32>, key_value: &Array2<f32>) -> Array2<f32> {
        // Simplified cross-attention
        query.dot(&self.cross_weights).dot(&key_value.t()).dot(key_value)
    }

    fn combine_streams(&self, algebra: &Array2<f32>, geometry: &Array2<f32>) -> Array2<f32> {
        // Simple combination
        (algebra + geometry) / 2.0
    }

    fn compute_attention_weights(&self, trees: &[Tree]) -> HashMap<String, Array2<f32>> {
        let mut weights = HashMap::new();
        let batch_size = trees.len();
        weights.insert("algebra".to_string(), Array2::eye(batch_size));
        weights.insert("geometry".to_string(), Array2::eye(batch_size));
        weights
    }
}

/// Output from dual-stream transformer
pub struct DualStreamOutput {
    pub algebra_features: Array2<f32>,
    pub geometry_features: Array2<f32>,
    pub combined_features: Array2<f32>,
    pub attention_weights: HashMap<String, Array2<f32>>,
}

/// Algebra stream processor
pub struct AlgebraStream {
    embed_dim: usize,
}

impl AlgebraStream {
    pub fn new(embed_dim: usize) -> Self {
        AlgebraStream { embed_dim }
    }

    pub fn process(&self, tree: &Tree) -> Array1<f32> {
        // Simple feature extraction
        let mut features = Array1::zeros(self.embed_dim);
        features[0] = tree.size() as f32;
        if self.embed_dim > 1 {
            features[1] = tree.coproduct().len() as f32;
        }
        features
    }
}

/// Geometry stream processor
pub struct GeometryStream {
    embed_dim: usize,
}

impl GeometryStream {
    pub fn new(embed_dim: usize) -> Self {
        GeometryStream { embed_dim }
    }

    pub fn embed(&self, tree: &Tree) -> Array1<f32> {
        // Simple geometric embedding
        let mut coords = Array1::zeros(self.embed_dim);
        coords[0] = tree.size() as f32;
        if self.embed_dim > 1 {
            coords[1] = tree.max_depth() as f32;
        }
        if self.embed_dim > 2 {
            coords[2] = tree.leaf_count() as f32;
        }
        coords
    }
}

/// Trainer combining Hopf-invariant objectives with the dual-stream transformer
pub struct DualStreamTrainer {
    transformer: DualStreamTransformer,
    hopf_loss: HopfInvariantLoss,
    algebra_stream: AlgebraStream,
    geometry_stream: GeometryStream,
    embedding: Box<dyn GeometricEmbedding>,
}

impl DualStreamTrainer {
    /// Create a new trainer
    pub fn new(
        transformer: DualStreamTransformer,
        embedding: Box<dyn GeometricEmbedding>,
    ) -> Self {
        let embed_dim = transformer.embed_dim;
        let algebra_stream = AlgebraStream::new(embed_dim);
        let geometry_stream = GeometryStream::new(embed_dim);
        let hopf_loss = HopfInvariantLoss::new(AlgebraicConstraints::default());

        DualStreamTrainer {
            transformer,
            hopf_loss,
            algebra_stream,
            geometry_stream,
            embedding,
        }
    }

    /// Create a trainer directly from a [`DualStreamConfig`]
    pub fn from_config(cfg: DualStreamConfig, embedder: Box<dyn GeometricEmbedding>) -> Self {
        let transformer = DualStreamTransformer::new(cfg);
        Self::new(transformer, embedder)
    }

    /// Perform a single training step and return the Hopf-invariant loss
    pub fn train_step(&mut self, trees: &[Tree]) -> f32 {
        let mut algebra_embeds = Array2::zeros((trees.len(), self.transformer.embed_dim));
        let mut geometry_embeds = Array2::zeros((trees.len(), self.transformer.embed_dim));

        for (i, tree) in trees.iter().enumerate() {
            algebra_embeds.row_mut(i).assign(&self.algebra_stream.process(tree));
            let geom = self.embedding.embed(tree);
            for j in 0..self.transformer.embed_dim.min(geom.len()) {
                geometry_embeds[(i, j)] = geom[j];
            }
        }

        let output = self.transformer.forward(trees, algebra_embeds, geometry_embeds);

        let mut embed_map: HashMap<Tree, Array1<f32>> = HashMap::new();
        for (i, tree) in trees.iter().enumerate() {
            embed_map.insert(tree.clone(), output.combined_features.row(i).to_owned());
        }

        let pairs: Vec<(Tree, Tree)> = trees.iter().map(|t| (t.clone(), t.clone())).collect();
        self.hopf_loss.total_loss(&pairs, |t| {
            embed_map
                .get(t)
                .cloned()
                .unwrap_or_else(|| self.embedding.embed(t))
        })
    }

    /// Train for multiple epochs, returning the loss history
    pub fn train_epochs(&mut self, trees: &[Tree], epochs: usize) -> Vec<f32> {
        let mut history = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            history.push(self.train_step(trees));
        }
        history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::TreeBuilder;

    struct DummyEmbedding;

    impl GeometricEmbedding for DummyEmbedding {
        fn embed(&self, tree: &Tree) -> Array1<f32> {
            Array1::from_vec(vec![tree.size() as f32, tree.max_depth() as f32])
        }

        fn distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
            (a - b).mapv(|x| x * x).sum().sqrt()
        }

        fn interpolate(&self, a: &Array1<f32>, b: &Array1<f32>, t: f32) -> Array1<f32> {
            a * (1.0 - t) + b * t
        }
    }

    #[test]
    fn test_dual_stream_forward() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1);
        let tree = builder.build().unwrap();

        let cfg = DualStreamConfig {
            embed_dim: 4,
            n_heads: 2,
            algebra_attention: AttentionMechanism::Standard,
            geometry_attention: AttentionMechanism::Standard,
        };
        let transformer = DualStreamTransformer::new(cfg);
        let algebra = AlgebraStream::new(4);
        let geometry = GeometryStream::new(4);

        let output = transformer.forward_with_streams(&[tree], &algebra, &geometry);
        assert_eq!(output.combined_features.nrows(), 1);
        assert_eq!(output.combined_features.ncols(), 4);
    }

    #[test]
    fn test_dual_stream_trainer() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1);
        let t1 = builder.build().unwrap();
        let t2 = Tree::new();

        let cfg = DualStreamConfig {
            embed_dim: 4,
            n_heads: 1,
            algebra_attention: AttentionMechanism::Standard,
            geometry_attention: AttentionMechanism::Standard,
        };
        let transformer = DualStreamTransformer::new(cfg);
        let embedder = Box::new(DummyEmbedding);
        let mut trainer = DualStreamTrainer::new(transformer, embedder);

        let loss = trainer.train_step(&[t1, t2]);
        assert!(loss >= 0.0);
    }
}
