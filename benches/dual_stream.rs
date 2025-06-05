use criterion::{criterion_group, criterion_main, Criterion};
use hopf_ml::applications::{DualStreamTransformer, DualStreamTrainer, DualStreamConfig, AttentionMechanism};
use hopf_ml::core::{HopfML, GeometricHopfFlow};

fn bench_train_step(c: &mut Criterion) {
    let mut hopf = HopfML::new();
    let trees = hopf.get_trees(3).clone();
    let cfg = DualStreamConfig {
        embed_dim: 8,
        n_heads: 2,
        algebra_attention: AttentionMechanism::Standard,
        geometry_attention: AttentionMechanism::Standard,
    };
    let transformer = DualStreamTransformer::new(cfg);
    let embedder = Box::new(GeometricHopfFlow::new(8));
    let mut trainer = DualStreamTrainer::new(transformer, embedder);

    c.bench_function("dual_stream_train_step", |b| {
        b.iter(|| {
            trainer.train_step(&trees);
        })
    });
}

criterion_group!(benches, bench_train_step);
criterion_main!(benches);
