//! Edge classification module for tree edge predictions

use std::sync::Arc;
use ndarray::{Array1, Array2, Axis};
use crate::algebra::{Tree, Forest};
use crate::graph::{GraphData, EdgeIndex};

/// Types of edge classifications
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeClass {
    /// Edge that would be cut in an admissible cut
    Cuttable,
    /// Edge that preserves tree structure
    Structural,
    /// Edge involved in antipode transformation
    AntipodeRelevant,
    /// Edge in a grafting position
    GraftingSite,
}

/// Edge classifier for tree edges
pub struct EdgeClassifier {
    /// Number of edge features
    n_features: usize,
    /// Classification mode
    mode: ClassificationMode,
}

#[derive(Debug, Clone, Copy)]
pub enum ClassificationMode {
    /// Classify edges for admissible cuts
    AdmissibleCuts,
    /// Classify edges for antipode operations
    Antipode,
    /// Classify edges for grafting positions
    Grafting,
    /// Multi-task classification
    MultiTask,
}

impl EdgeClassifier {
    /// Create a new edge classifier
    pub fn new(mode: ClassificationMode) -> Self {
        EdgeClassifier {
            n_features: 16, // Will compute 16 edge features
            mode,
        }
    }
    
    /// Extract features for all edges in a tree
    pub fn extract_edge_features(&self, tree: &Tree) -> Array2<f32> {
        let n_edges = tree.size() - 1; // Tree has n-1 edges
        let mut features = Array2::zeros((n_edges, self.n_features));
        
        // Build edge list (parent -> child edges)
        let mut edge_idx = 0;
        for parent in 0..tree.size() {
            for &child in tree.children(parent) {
                self.compute_edge_features(tree, parent, child, features.row_mut(edge_idx));
                edge_idx += 1;
            }
        }
        
        features
    }
    
    /// Compute features for a single edge
    fn compute_edge_features(
        &self,
        tree: &Tree,
        parent: usize,
        child: usize,
        mut features: ndarray::ArrayViewMut1<f32>,
    ) {
        // Basic edge features
        features[0] = tree.node_depth(parent) as f32;
        features[1] = tree.node_depth(child) as f32;
        features[2] = tree.subtree_size(parent) as f32;
        features[3] = tree.subtree_size(child) as f32;
        features[4] = tree.node_degree(parent) as f32;
        features[5] = tree.node_degree(child) as f32;
        
        // Relative features
        features[6] = (tree.subtree_size(child) as f32) / (tree.size() as f32);
        features[7] = if tree.node_degree(parent) > 0 {
            1.0 / tree.node_degree(parent) as f32
        } else {
            0.0
        };
        
        // Structural features
        features[8] = if tree.node_degree(child) == 0 { 1.0 } else { 0.0 }; // Is leaf
        features[9] = if parent == 0 { 1.0 } else { 0.0 }; // From root
        
        // Path features
        let path_to_root = tree.path_to_root(child);
        features[10] = path_to_root.len() as f32;
        
        // Sibling features
        let siblings = tree.children(parent);
        features[11] = siblings.len() as f32;
        let sibling_sizes: Vec<usize> = siblings.iter()
            .map(|&s| tree.subtree_size(s))
            .collect();
        features[12] = if !sibling_sizes.is_empty() {
            sibling_sizes.iter().sum::<usize>() as f32 / sibling_sizes.len() as f32
        } else {
            0.0
        };
        
        // Balance features
        let left_size = tree.children(parent)
            .iter()
            .take_while(|&&n| n < child)
            .map(|&n| tree.subtree_size(n))
            .sum::<usize>();
        let right_size = tree.children(parent)
            .iter()
            .skip_while(|&&n| n <= child)
            .map(|&n| tree.subtree_size(n))
            .sum::<usize>();
        features[13] = left_size as f32;
        features[14] = right_size as f32;
        features[15] = (left_size as f32 - right_size as f32).abs();
    }
    
    /// Create edge labels for admissible cuts
    pub fn label_edges_for_cuts(&self, tree: &Tree, cuts: &[Vec<usize>]) -> Array1<f32> {
        let n_edges = tree.size() - 1;
        let mut labels = Array1::zeros(n_edges);
        
        // For each cut, mark edges that are cut
        for cut in cuts {
            let mut edge_idx = 0;
            for parent in 0..tree.size() {
                for &child in tree.children(parent) {
                    // Edge is cut if child is in cut but parent is not
                    if cut.contains(&child) && !cut.contains(&parent) {
                        labels[edge_idx] = 1.0;
                    }
                    edge_idx += 1;
                }
            }
        }
        
        labels
    }
    
    /// Create edge labels for grafting positions
    pub fn label_edges_for_grafting(&self, tree: &Tree) -> Array1<f32> {
        let n_edges = tree.size() - 1;
        let mut labels = Array1::zeros(n_edges);
        
        // All edges can potentially be grafting sites
        // But leaf edges are more likely
        let mut edge_idx = 0;
        for parent in 0..tree.size() {
            for &child in tree.children(parent) {
                if tree.children(child).is_empty() {
                    labels[edge_idx] = 1.0; // Leaf edge
                } else {
                    labels[edge_idx] = 0.5; // Internal edge
                }
                edge_idx += 1;
            }
        }
        
        labels
    }
    
    /// Get edge predictions as node pairs
    pub fn edges_from_predictions(
        &self,
        tree: &Tree,
        predictions: &Array1<f32>,
        threshold: f32,
    ) -> Vec<(usize, usize)> {
        let mut selected_edges = Vec::new();
        let mut edge_idx = 0;
        
        for parent in 0..tree.size() {
            for &child in tree.children(parent) {
                if predictions[edge_idx] > threshold {
                    selected_edges.push((parent, child));
                }
                edge_idx += 1;
            }
        }
        
        selected_edges
    }
    
    /// Convert edge predictions to admissible cuts
    pub fn edges_to_cuts(
        &self,
        tree: &Tree,
        edge_predictions: &[(usize, usize)],
    ) -> Vec<Vec<usize>> {
        let mut cuts = Vec::new();
        
        // Group edges by connected components
        let mut visited = vec![false; tree.size()];
        let mut current_cut = Vec::new();
        
        for &(parent, child) in edge_predictions {
            if !visited[child] {
                current_cut.push(child);
                visited[child] = true;
            }
        }
        
        if !current_cut.is_empty() {
            cuts.push(current_cut);
        }
        
        cuts
    }
}

/// Edge attention mechanism for tree transformers
pub struct EdgeAttention {
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of attention heads
    n_heads: usize,
}

impl EdgeAttention {
    /// Create new edge attention module
    pub fn new(hidden_dim: usize, n_heads: usize) -> Self {
        EdgeAttention { hidden_dim, n_heads }
    }
    
    /// Compute edge-aware attention scores
    pub fn compute_attention(
        &self,
        node_features: &Array2<f32>,
        edge_features: &Array2<f32>,
        adjacency: &[(usize, usize)],
    ) -> Array2<f32> {
        let n_nodes = node_features.nrows();
        let mut attention = Array2::zeros((n_nodes, n_nodes));
        
        // For each edge, compute attention score
        for (edge_idx, &(src, dst)) in adjacency.iter().enumerate() {
            let src_feat = node_features.row(src);
            let dst_feat = node_features.row(dst);
            let edge_feat = edge_features.row(edge_idx);
            
            // Simple dot-product attention with edge features
            let score = src_feat.dot(&dst_feat) + edge_feat.sum();
            attention[(src, dst)] = score;
        }
        
        // Normalize attention scores
        for i in 0..n_nodes {
            let row_sum: f32 = attention.row(i).sum();
            if row_sum > 0.0 {
                attention.row_mut(i).mapv_inplace(|x| x / row_sum);
            }
        }
        
        attention
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::TreeBuilder;
    
    #[test]
    fn test_edge_features() {
        let tree = TreeBuilder::new()
            .add_child(0, 1)
            .add_child(0, 2)
            .add_child(1, 3)
            .build()
            .unwrap();
            
        let classifier = EdgeClassifier::new(ClassificationMode::AdmissibleCuts);
        let features = classifier.extract_edge_features(&tree);
        
        assert_eq!(features.nrows(), 3); // 3 edges in the tree
        assert_eq!(features.ncols(), 16); // 16 features per edge
    }
    
    #[test]
    fn test_edge_labeling() {
        let tree = TreeBuilder::new()
            .add_child(0, 1)
            .add_child(0, 2)
            .build()
            .unwrap();
            
        let classifier = EdgeClassifier::new(ClassificationMode::AdmissibleCuts);
        let cuts = vec![vec![1], vec![2], vec![1, 2]];
        let labels = classifier.label_edges_for_cuts(&tree, &cuts);
        
        assert_eq!(labels.len(), 2); // 2 edges
        assert!(labels.iter().any(|&x| x > 0.0)); // At least one edge is cut
    }
} 