//! Advanced features demonstration

use hopf_ml::algebra::{Tree, TreeBuilder};
use hopf_ml::applications::{
    DualStreamTransformer, AttentionMechanism, AlgebraStream, GeometryStream,
    DualStreamConfig,
    HopfChain, HopfContract, HopfOperation, AlgebraicTransaction, AlgebraicObject,
    QuantumTreeState, HopfGate, HopfQuantumCircuit, HopfVQE, TreeObservable,
    ComputationProof,
};
use hopf_ml::{Antipode, CoProduct};
use hopf_ml::core::{
    HopfInvariantLoss, AlgebraicConstraints, HopfFlow, GeometricHopfFlow,
    HyperbolicEmbedding, SphericalEmbedding, CGA, GeometricEmbedding,
};
use ndarray::{Array1, Array2};

fn main() {
    println!("=== Hopf-ML Advanced Features Demo ===\n");

    // Create some test trees
    let trees = vec![
        TreeBuilder::new().build().unwrap(),
        {
            let mut b = TreeBuilder::new();
            b.add_child(0, 1);
            b.build().unwrap()
        },
        {
            let mut b = TreeBuilder::new();
            b.add_child(0, 1);
            b.add_child(0, 2);
            b.build().unwrap()
        },
        {
            let mut b = TreeBuilder::new();
            b.add_child(0, 1);
            b.add_child(1, 2);
            b.add_child(1, 3);
            b.build().unwrap()
        },
    ];

    // 1. Hopf-Invariant Loss Functions
    println!("1. Hopf-Invariant Loss Functions");
    demo_hopf_invariant_loss(&trees);

    // 2. Dual-Stream Transformer Architecture
    println!("\n2. Dual-Stream Transformer Architecture");
    demo_dual_stream_transformer(&trees);

    // 3. Neural Differential Hopf Flows
    println!("\n3. Neural Differential Hopf Flows");
    demo_neural_hopf_flows(&trees[2]);

    // 4. Geometric Embeddings
    println!("\n4. Geometric Embeddings (Hyperbolic, Spherical, CGA)");
    demo_geometric_embeddings(&trees[1]);

    // 5. Blockchain Integration
    println!("\n5. Blockchain Integration");
    demo_blockchain(&trees);

    // 6. Quantum-Inspired Approaches
    println!("\n6. Quantum-Inspired Approaches");
    demo_quantum(&trees);
}

fn demo_hopf_invariant_loss(trees: &[Tree]) {
    let constraints = AlgebraicConstraints {
        product_weight: 1.0,
        antipode_weight: 2.0,
        coproduct_weight: 0.5,
        temperature: 0.1,
    };

    let mut loss_calc = HopfInvariantLoss::new(constraints);
    
    // Simple embedding function
    let embed_fn = |tree: &Tree| -> Array1<f32> {
        let mut embed = Array1::zeros(10);
        embed[0] = tree.size() as f32;
        embed[1] = tree.coproduct().len() as f32;
        embed
    };

    // Calculate losses
    if trees.len() >= 2 {
        let prod_loss = loss_calc.product_linearity_loss(&trees[0], &trees[1], embed_fn);
        let anti_loss = loss_calc.antipode_equivariance_loss(&trees[1], embed_fn);
        let cop_loss = loss_calc.coproduct_contrastive_loss(&trees[2], embed_fn);

        println!("  Product linearity loss: {:.4}", prod_loss);
        println!("  Antipode equivariance loss: {:.4}", anti_loss);
        println!("  Coproduct contrastive loss: {:.4}", cop_loss);
    }
}

fn demo_dual_stream_transformer(trees: &[Tree]) {
    let embed_dim = 16;
    let n_heads = 4;

    // Create transformer
    let cfg = DualStreamConfig {
        embed_dim,
        n_heads,
        algebra_attention: AttentionMechanism::AlgebraWeighted,
        geometry_attention: AttentionMechanism::Hyperbolic,
    };
    let transformer = DualStreamTransformer::new(cfg);

    // Create algebra and geometry embeddings
    let algebra_stream = AlgebraStream::new(embed_dim);
    let geometry_stream = GeometryStream::new(embed_dim);

    let mut algebra_embeds = Array2::zeros((trees.len(), embed_dim));
    let mut geometry_embeds = Array2::zeros((trees.len(), embed_dim));

    for (i, tree) in trees.iter().enumerate() {
        let alg_features = algebra_stream.process(tree);
        let geo_features = geometry_stream.embed(tree);
        
        algebra_embeds.slice_mut(ndarray::s![i, ..]).assign(&alg_features);
        geometry_embeds.slice_mut(ndarray::s![i, ..]).assign(&geo_features);
    }

    // Forward pass
    let output = transformer.forward(trees, algebra_embeds, geometry_embeds);

    println!("  Algebra features shape: {:?}", output.algebra_features.dim());
    println!("  Geometry features shape: {:?}", output.geometry_features.dim());
    println!("  Combined features shape: {:?}", output.combined_features.dim());
    println!("  Attention weights: {} types", output.attention_weights.len());
}

fn demo_neural_hopf_flows(tree: &Tree) {
    // Create Hopf flow
    let mut hopf_flow = HopfFlow::new(20);
    
    // Simulate coproduct flow
    let flow_result = hopf_flow.coproduct_flow(tree, 10);
    
    println!("  Flow trajectory length: {}", flow_result.trajectory.len());
    println!("  Forest embedding dim: {}", flow_result.forest_embedding.len());
    println!("  Trunk embedding dim: {}", flow_result.trunk_embedding.len());
    println!("  Total flow time: {:.2}", flow_result.flow_time);

    // Train one step
    let loss = hopf_flow.train_step(tree, 0.01);
    println!("  Training loss: {:.4}", loss);

    // Geometric flow
    let geo_flow = GeometricHopfFlow::new(10);
    let start = Array1::zeros(10);
    let velocity = Array1::ones(10);
    let endpoint = geo_flow.geodesic_flow(start, velocity, 1.0);
    println!("  Geodesic endpoint norm: {:.4}", endpoint.mapv(|x| x * x).sum().sqrt());
}

fn demo_geometric_embeddings(tree: &Tree) {
    // Hyperbolic embedding
    let hyp = HyperbolicEmbedding::new(3, -1.0);
    let hyp_embed = hyp.embed(tree);
    let hyp_norm = hyp_embed.mapv(|x| x * x).sum().sqrt();
    println!("  Hyperbolic embedding (Poincaré): norm = {:.4}", hyp_norm);

    // Spherical embedding
    let sphere = SphericalEmbedding::new(2);
    let sphere_embed = sphere.embed(tree);
    let sphere_norm = sphere_embed.mapv(|x| x * x).sum().sqrt();
    println!("  Spherical embedding: norm = {:.4}", sphere_norm);

    // Conformal Geometric Algebra
    let cga = CGA::new(3);
    let cga_multivector = cga.embed(tree);
    println!("  CGA embedding: grade = {}, dim = {}", 
        cga_multivector.grade, cga_multivector.coefficients.len());

    // Test interpolation
    let tree2 = {
        let mut b = TreeBuilder::new();
        b.add_child(0, 1);
        b.add_child(0, 2);
        b.build().unwrap()
    };
    let hyp_embed2 = hyp.embed(&tree2);
    let interpolated = hyp.interpolate(&hyp_embed, &hyp_embed2, 0.5);
    let interp_norm = interpolated.mapv(|x| x * x).sum().sqrt();
    println!("  Interpolated embedding norm: {:.4}", interp_norm);
}

fn demo_blockchain(trees: &[Tree]) {
    // Create blockchain
    let mut chain = HopfChain::new(2);
    println!("  Created blockchain with difficulty 2");

    // Add transactions
    for (i, tree) in trees.iter().take(2).enumerate() {
        let proof = ComputationProof::new(
            "proof_hash".to_string(),
            vec![],
            "verifier".to_string(),
        );
        let tx = AlgebraicTransaction::new(
            format!("tx_{}", i),
            HopfOperation::Antipode,
            AlgebraicObject::Tree(tree.clone()),
            AlgebraicObject::Forest(tree.antipode()),
            proof,
            i as u64,
        );

        chain.add_transaction(tx).unwrap();
    }

    // Mine block
    let block = chain.mine_block("miner_address");
    println!("  Mined block: {:?}", block);

    // Validate chain
    println!("  Chain valid: {}", chain.is_valid_chain());

    // Smart contract
    let mut contract = HopfContract::new();
    let result = contract.execute(HopfOperation::NaturalGrowth, trees[0].clone());
    match result {
        Ok(AlgebraicObject::Forest(forest)) => {
            println!("  Contract executed: {} output trees", forest.trees().len());
        }
        _ => println!("  Contract execution failed"),
    }
}

fn demo_quantum(trees: &[Tree]) {
    // Create quantum state
    let basis = trees[..2].to_vec();
    let mut state = QuantumTreeState::new(basis);
    println!("  Created quantum state with {} basis trees", trees[..2].len());

    // Apply gates
    state.apply_hopf_gate(&HopfGate::Hadamard);
    state.apply_hopf_gate(&HopfGate::EntangleGate(0.25));
    state.apply_hopf_gate(&HopfGate::AntipodeGate);

    // Measure
    let (measured_tree, prob) = state.measure();
    println!("  Measured tree size: {}, probability: {:.4}", measured_tree.size(), prob);

    // Quantum circuit
    let circuit = HopfQuantumCircuit::qft_circuit(2);
    let initial = QuantumTreeState::from_tree(trees[1].clone());
    let final_state = circuit.simulate(initial);
    
    let observable = SizeObservable;
    let expectation = final_state.expectation_value(&observable);
    println!("  QFT expectation value of size: {:.4}", expectation);

    // VQE
    let mut vqe = HopfVQE::new(Box::new(SizeObservable));
    let initial_vqe = QuantumTreeState::from_tree(trees[0].clone());
    let energies = vqe.optimize(initial_vqe, 3);
    
    println!("  VQE optimization:");
    for (i, energy) in energies.iter().enumerate() {
        println!("    Iteration {}: energy = {:.4}", i, energy);
    }
}

// Implement required trait for the example
impl TreeObservable for SizeObservable {
    fn eigenvalue(&self, tree: &Tree) -> f64 {
        tree.size() as f64
    }
    
    fn matrix_representation(&self, basis: &[Tree]) -> Option<Array2<num_complex::Complex64>> {
        use num_complex::Complex64;
        let n = basis.len();
        let mut matrix = Array2::zeros((n, n));
        
        for i in 0..n {
            matrix[[i, i]] = Complex64::new(self.eigenvalue(&basis[i]), 0.0);
        }
        
        Some(matrix)
    }
}

pub struct SizeObservable; 