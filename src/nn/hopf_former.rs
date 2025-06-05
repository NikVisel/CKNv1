//! HopfFormer: multi-layer dual-stream transformer architecture
//! Combining algebraic and geometric embeddings with cross-attention
//! and feed-forward residual blocks.

use crate::algebra::Tree;
use crate::core::{GeometricEmbedding, HopfInvariantLoss, AlgebraicConstraints};
use crate::applications::{DualStreamTransformer, AttentionMechanism, AlgebraStream, GeometryStream};
use ndarray::{Array1, Array2};

/// Configuration for HopfFormer
#[derive(Debug, Clone)]
pub struct HopfFormerConfig {
    /// Dimension of embeddings
    pub embed_dim: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Number of attention heads per layer
    pub n_heads: usize,
}

impl Default for HopfFormerConfig {
    fn default() -> Self {
        HopfFormerConfig { embed_dim: 64, n_layers: 4, n_heads: 4 }
    }
}

/// HopfFormer layer built on DualStreamTransformer
struct HopfFormerLayer {
    transformer: DualStreamTransformer,
}

impl HopfFormerLayer {
    fn new(cfg: &HopfFormerConfig) -> Self {
        let transformer = DualStreamTransformer::new(
            cfg.embed_dim,
            cfg.n_heads,
            AttentionMechanism::AlgebraWeighted,
            AttentionMechanism::Hyperbolic,
        );
        HopfFormerLayer { transformer }
    }

    fn forward(&self, trees: &[Tree], algebra: Array2<f32>, geom: Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let out = self.transformer.forward(trees, algebra, geom);
        (out.algebra_features, out.geometry_features)
    }
}

/// HopfFormer architecture stacking multiple HopfFormerLayers
pub struct HopfFormer {
    layers: Vec<HopfFormerLayer>,
    hopf_loss: HopfInvariantLoss,
    algebra_stream: AlgebraStream,
    geometry_stream: GeometryStream,
    embed_dim: usize,
    embedder: Box<dyn GeometricEmbedding>,
}

impl HopfFormer {
    /// Build a new HopfFormer with the given configuration and embedding
    pub fn new(cfg: HopfFormerConfig, embedder: Box<dyn GeometricEmbedding>) -> Self {
        let layers = (0..cfg.n_layers).map(|_| HopfFormerLayer::new(&cfg)).collect();
        let algebra_stream = AlgebraStream::new(cfg.embed_dim);
        let geometry_stream = GeometryStream::new(cfg.embed_dim);
        let hopf_loss = HopfInvariantLoss::new(AlgebraicConstraints::default());
        HopfFormer {
            layers,
            hopf_loss,
            algebra_stream,
            geometry_stream,
            embed_dim: cfg.embed_dim,
            embedder,
        }
    }

    /// Forward pass returning the final embeddings for each tree
    pub fn forward(&self, trees: &[Tree]) -> Array2<f32> {
        let batch = trees.len();
        let mut algebra = Array2::zeros((batch, self.embed_dim));
        let mut geom = Array2::zeros((batch, self.embed_dim));
        for (i, tree) in trees.iter().enumerate() {
            algebra.row_mut(i).assign(&self.algebra_stream.process(tree));
            geom.row_mut(i).assign(&self.geometry_stream.embed(tree));
        }
        for layer in &self.layers {
            let (a, g) = layer.forward(trees, algebra.clone(), geom.clone());
            algebra = a;
            geom = g;
        }
        (algebra + geom) / 2.0
    }

    /// Compute Hopf invariant loss for a batch of trees
    pub fn hopf_loss(&mut self, trees: &[Tree]) -> f32 {
        let embeddings = self.forward(trees);
        let mut map = std::collections::HashMap::new();
        for (i, tree) in trees.iter().enumerate() {
            map.insert(tree.clone(), embeddings.row(i).to_owned());
        }
        let pairs: Vec<(Tree, Tree)> = trees.iter().map(|t| (t.clone(), t.clone())).collect();
        self.hopf_loss.total_loss(&pairs, |t| map.get(t).cloned().unwrap_or_else(|| self.embedder.embed(t)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::TreeBuilder;

    struct DummyEmbed;
    impl GeometricEmbedding for DummyEmbed {
        fn embed(&self, tree: &Tree) -> Array1<f32> { Array1::from(vec![tree.size() as f32]) }
        fn distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 { (a[0]-b[0]).abs() }
        fn interpolate(&self, a: &Array1<f32>, b: &Array1<f32>, t: f32) -> Array1<f32> { a*(1.0-t)+b*t }
    }

    #[test]
    fn test_hopf_former_forward() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0,1);
        let tree = builder.build().unwrap();
        let cfg = HopfFormerConfig::default();
        let model = HopfFormer::new(cfg, Box::new(DummyEmbed));
        let out = model.forward(&[tree]);
        assert_eq!(out.nrows(), 1);
    }
}
