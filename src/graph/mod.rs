//! Graph representations of trees for neural networks

mod graph_data;
mod features;
mod conversion;

use crate::algebra::Tree;
use ndarray::Array2;
use std::collections::VecDeque;

pub use graph_data::{GraphData, EdgeIndex};
pub use features::{NodeFeatures, FeatureExtractor, StandardFeatureExtractor};
pub use conversion::{tree_to_graph, TreeGraph};

/// Extract node features from tree structure
pub fn extract_node_features(tree: &Tree) -> Array2<f32> {
    let n_nodes = tree.size();
    let mut features = Array2::zeros((n_nodes, 12)); // Increased feature dimensions
    
    // Basic features (keep existing)
    for node in 0..n_nodes {
        features[(node, 0)] = tree.children(node).len() as f32;
        features[(node, 1)] = tree.node_depth(node) as f32;
        features[(node, 2)] = if tree.children(node).is_empty() { 1.0 } else { 0.0 };
        features[(node, 3)] = if node == 0 { 1.0 } else { 0.0 };
        
        // Enhanced features using new tree methods
        features[(node, 4)] = tree.subtree_size(node) as f32;
        features[(node, 5)] = tree.path_to_root(node).len() as f32;
        
        // Relative position features
        let max_depth = tree.max_depth() as f32;
        features[(node, 6)] = if max_depth > 0.0 { 
            tree.node_depth(node) as f32 / max_depth 
        } else { 
            0.0 
        };
        
        // Sibling information
        let siblings_count = if let Some(parent) = tree.parent(node) {
            tree.children(parent).len() - 1
        } else {
            0
        };
        features[(node, 7)] = siblings_count as f32;
        
        // Position among siblings (birth order)
        let sibling_position = if let Some(parent) = tree.parent(node) {
            tree.children(parent).iter().position(|&n| n == node).unwrap_or(0)
        } else {
            0
        };
        features[(node, 8)] = sibling_position as f32;
        
        // Distance to nearest leaf
        let min_leaf_dist = min_distance_to_leaf(tree, node);
        features[(node, 9)] = min_leaf_dist as f32;
        
        // Branching factor statistics in subtree
        let (mean_bf, max_bf) = subtree_branching_stats(tree, node);
        features[(node, 10)] = mean_bf;
        features[(node, 11)] = max_bf as f32;
    }
    
    features
}

/// Compute minimum distance from node to any leaf in its subtree
fn min_distance_to_leaf(tree: &Tree, node: usize) -> usize {
    if tree.children(node).is_empty() {
        0
    } else {
        tree.children(node)
            .iter()
            .map(|&child| 1 + min_distance_to_leaf(tree, child))
            .min()
            .unwrap_or(0)
    }
}

/// Compute branching factor statistics for a subtree
fn subtree_branching_stats(tree: &Tree, node: usize) -> (f32, usize) {
    let mut queue = VecDeque::new();
    let mut branching_factors = Vec::new();
    queue.push_back(node);
    
    while let Some(n) = queue.pop_front() {
        let degree = tree.children(n).len();
        if degree > 0 {
            branching_factors.push(degree);
            for &child in tree.children(n) {
                queue.push_back(child);
            }
        }
    }
    
    if branching_factors.is_empty() {
        (0.0, 0)
    } else {
        let mean = branching_factors.iter().sum::<usize>() as f32 
            / branching_factors.len() as f32;
        let max = *branching_factors.iter().max().unwrap_or(&0);
        (mean, max)
    }
} 