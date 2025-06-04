//! Basic example of Hopf algebra operations on rooted trees

use hopf_ml::prelude::*;
use hopf_ml::utils::timing::Timer;

fn main() {
    println!("=== Hopf Algebra of Rooted Trees ===\n");

    // Create some trees
    let t1 = Tree::new(); // Single node
    let t2 = TreeBuilder::new()
        .add_child(0, 1)
        .build()
        .unwrap();
    let t3 = TreeBuilder::new()
        .add_child(0, 1)
        .add_child(0, 2)
        .build()
        .unwrap();

    println!("Tree 1 (single node):");
    println!("{:?}", t1);
    
    println!("\nTree 2 (two nodes):");
    println!("{:?}", t2);
    
    println!("\nTree 3 (three nodes, Y-shape):");
    println!("{:?}", t3);

    // Demonstrate coproduct
    println!("\n--- Coproduct ---");
    {
        let _timer = Timer::new("Coproduct computation");
        let cop = t3.coproduct();
        println!("Δ(t3) has {} terms:", cop.len());
        
        for ((forest, trunk), coeff) in cop.iter() {
            println!("  {} * ({:?} ⊗ {:?})", coeff, forest, trunk);
        }
    }

    // Demonstrate antipode
    println!("\n--- Antipode ---");
    {
        let _timer = Timer::new("Antipode computation");
        let s_t2 = t2.antipode();
        let s_t3 = t3.antipode();
        
        println!("S(t2) = {:?}", s_t2);
        println!("S(t3) = {:?}", s_t3);
        
        // Verify involution property
        let s_s_t2 = s_t2.antipode();
        println!("\nVerifying S(S(t2)) = t2:");
        println!("S(S(t2)) = {:?}", s_s_t2);
        println!("Equal to t2? {}", s_s_t2.trees().len() == 1 && s_s_t2.trees()[0] == t2);
    }

    // Demonstrate grafting (natural growth)
    println!("\n--- Natural Growth (Grafting) ---");
    let grafted = t2.graft_all_leaves();
    println!("Grafting t2 produces {} trees:", grafted.len());
    for (i, tree) in grafted.iter().enumerate() {
        println!("  Grafted tree {}: size = {}", i + 1, tree.size());
    }

    // Generate delta_k
    println!("\n--- Delta Generation ---");
    use hopf_ml::algebra::delta_k;
    for k in 1..=5 {
        let trees = delta_k(k);
        println!("δ_{} = sum of {} trees of size {}", k, trees.len(), k);
    }

    // Convert to graph data for ML
    println!("\n--- Graph Conversion ---");
    let graph = tree_to_graph(&t3);
    println!("Tree t3 as graph:");
    println!("  Nodes: {}", graph.num_nodes);
    println!("  Edges: {}", graph.edge_index.num_edges());
    println!("  Feature dimension: {}", graph.feature_dim());
    println!("  Node features shape: {:?}", graph.node_features.shape());

    // Demonstrate HopfML integration
    println!("\n--- HopfML Integration ---");
    let mut hopf_ml = HopfML::new();
    
    let stats = hopf_ml.compute_statistics(4);
    println!("Statistics for trees of size 4:");
    println!("  Count: {}", stats.count);
    println!("  Average cuts: {:.2}", stats.avg_cuts);
    println!("  Max cuts: {}", stats.max_cuts);
    println!("  Average height: {:.2}", stats.avg_height);
    println!("  Average leaves: {:.2}", stats.avg_leaves);

    println!("\n=== Example Complete ===");
} 