use hopf_ml::applications::{
    DualStreamTransformer, AttentionMechanism, DualStreamTrainer, DualStreamConfig,
};
use hopf_ml::core::{HopfML, GeometricHopfFlow};

fn main() {
    // Initialize HopfML helper and gather small trees
    let mut hopf = HopfML::new();
    let trees = hopf.get_trees(3).clone();

    // Build dual-stream transformer
    let config = DualStreamConfig {
        embed_dim: 8,
        n_heads: 2,
        algebra_attention: AttentionMechanism::Standard,
        geometry_attention: AttentionMechanism::Standard,
    };
    let transformer = DualStreamTransformer::new(config);
    let embedder = Box::new(GeometricHopfFlow::new(8));
    let mut trainer = DualStreamTrainer::new(transformer, embedder);

    // Train for a few epochs over the small dataset
    let losses = trainer.train_epochs(&trees, 3);
    for (epoch, loss) in losses.iter().enumerate() {
        println!("Epoch {}: loss = {:.4}", epoch, loss);
    }
}
