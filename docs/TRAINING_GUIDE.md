# Training Guide

This guide describes how to prepare datasets and run the provided training loops.

## Dataset Formatting

Datasets are generated directly from the algebra implementation using
`HopfML`. The helper functions produce graphs and labels for several tasks:

- `generate_cuts_dataset(max_size)` – pairs `GraphData` with the number of
  admissible cuts.
- `generate_antipode_dataset(max_size)` – pairs graphs with antipode forests.
- `generate_coproduct_dataset(max_size)` – triples of tree, cut and coefficient.

For custom training pipelines you can convert the datasets into arrays using the
`featurize` methods provided by the dataset types. This yields an array of tree
features, cut encodings (if applicable) and numerical targets suitable for
neural models.

## Running the Dual-Stream Training Loop

The example `train_toy_model.rs` demonstrates a minimal training loop for the
`DualStreamTransformer`:

```bash
cargo run --example train_toy_model
```

This builds a small tree corpus, initialises the transformer and runs a few
epochs via `DualStreamTrainer::train_epochs`. The trainer internally computes
algebraic and geometric embeddings and evaluates the `HopfInvariantLoss`.

## Benchmarks

Performance of the training step can be measured with Criterion by running

```bash
cargo bench
```

The benchmark located at `benches/dual_stream.rs` executes one training step and
reports throughput and timing statistics.
