use hopf_ml::applications::{
    DualStreamTransformer, AttentionMechanism, DualStreamTrainer,
};
use hopf_ml::core::{HopfML, GeometricHopfFlow};

fn main() {
    // Initialize HopfML helper and gather small trees
    let mut hopf = HopfML::new();
    let trees = hopf.get_trees(3).clone();

    // Build dual-stream transformer
    let transformer = DualStreamTransformer::new(
        8,
        2,
        AttentionMechanism::Standard,
        AttentionMechanism::Standard,
    );
    let embedder = Box::new(GeometricHopfFlow::new(8));
    let mut trainer = DualStreamTrainer::new(transformer, embedder);

    // Train for a few epochs over the small dataset
    for epoch in 0..3 {
        let loss = trainer.train_step(&trees);
        println!("Epoch {epoch}: loss = {loss:.4}");
    }
}
