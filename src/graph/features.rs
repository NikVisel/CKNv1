//! Feature extraction for tree nodes

use crate::algebra::Tree;
use ndarray::{Array1, Array2};
use std::collections::VecDeque;

/// Node features for a tree
#[derive(Clone, Debug)]
pub struct NodeFeatures {
    /// Is this the root node?
    pub is_root: bool,
    /// Node degree (number of children + 1 if has parent)
    pub degree: usize,
    /// Depth from root
    pub depth: usize,
    /// Number of nodes in subtree
    pub subtree_size: usize,
    /// Number of leaves in subtree
    pub num_leaves: usize,
    /// Height of subtree
    pub height: usize,
    /// Additional custom features
    pub custom: Vec<f32>,
}

impl NodeFeatures {
    /// Convert to feature vector
    pub fn to_vec(&self) -> Vec<f32> {
        let mut features = vec![
            if self.is_root { 1.0 } else { 0.0 },
            self.degree as f32,
            self.depth as f32,
            self.subtree_size as f32,
            self.num_leaves as f32,
            self.height as f32,
        ];
        features.extend(&self.custom);
        features
    }

    /// Get feature dimension
    pub fn dim(&self) -> usize {
        6 + self.custom.len()
    }
}

/// Trait for extracting features from trees
pub trait FeatureExtractor {
    /// Extract features for all nodes in a tree
    fn extract_features(&self, tree: &Tree) -> Vec<NodeFeatures>;
    
    /// Get the feature dimension
    fn feature_dim(&self) -> usize;
    
    /// Convert features to ndarray
    fn features_to_array(&self, features: &[NodeFeatures]) -> Array2<f32> {
        let n = features.len();
        let d = self.feature_dim();
        let mut arr = Array2::zeros((n, d));
        
        for (i, feat) in features.iter().enumerate() {
            let vec = feat.to_vec();
            for (j, &val) in vec.iter().enumerate() {
                arr[[i, j]] = val;
            }
        }
        
        arr
    }
}

/// Standard feature extractor with common tree features
pub struct StandardFeatureExtractor {
    /// Whether to include Hopf-algebraic features
    include_hopf_features: bool,
}

impl StandardFeatureExtractor {
    /// Create new extractor
    pub fn new() -> Self {
        StandardFeatureExtractor {
            include_hopf_features: false,
        }
    }

    /// Enable Hopf-algebraic features
    pub fn with_hopf_features(mut self) -> Self {
        self.include_hopf_features = true;
        self
    }

    /// Compute subtree statistics
    fn compute_subtree_stats(&self, tree: &Tree, node: usize) -> (usize, usize, usize) {
        let mut size = 1;
        let mut leaves = 0;
        let mut height = 0;
        
        let children = tree.children(node);
        if children.is_empty() {
            leaves = 1;
        } else {
            for &child in children {
                let (child_size, child_leaves, child_height) = 
                    self.compute_subtree_stats(tree, child);
                size += child_size;
                leaves += child_leaves;
                height = height.max(child_height + 1);
            }
        }
        
        (size, leaves, height)
    }
}

impl Default for StandardFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor for StandardFeatureExtractor {
    fn extract_features(&self, tree: &Tree) -> Vec<NodeFeatures> {
        let n = tree.size();
        let mut features = Vec::with_capacity(n);
        
        // Compute depths
        let depths = tree.depths();
        
        // Compute subtree statistics for each node
        let mut subtree_stats = vec![(0, 0, 0); n];
        for node in 0..n {
            subtree_stats[node] = self.compute_subtree_stats(tree, node);
        }
        
        // Extract features for each node
        for node in 0..n {
            let parent = tree.parent(node);
            let children = tree.children(node);
            let degree = children.len() + if parent.is_some() { 1 } else { 0 };
            
            let (subtree_size, num_leaves, height) = subtree_stats[node];
            
            let mut custom = Vec::new();
            
            if self.include_hopf_features {
                // Add Hopf-specific features
                // Number of possible cuts in subtree
                let num_edges_in_subtree = subtree_size.saturating_sub(1);
                custom.push(num_edges_in_subtree as f32);
                
                // Branching factor
                let branching = if height > 0 {
                    (subtree_size - 1) as f32 / height as f32
                } else {
                    0.0
                };
                custom.push(branching);
            }
            
            features.push(NodeFeatures {
                is_root: node == 0,
                degree,
                depth: depths[node],
                subtree_size,
                num_leaves,
                height,
                custom,
            });
        }
        
        features
    }
    
    fn feature_dim(&self) -> usize {
        if self.include_hopf_features {
            8
        } else {
            6
        }
    }
}

/// Extract edge features for a tree
pub fn extract_edge_features(tree: &Tree) -> Vec<(usize, usize, Vec<f32>)> {
    let mut edge_features = Vec::new();
    
    for parent in 0..tree.size() {
        for (child_idx, &child) in tree.children(parent).iter().enumerate() {
            let features = vec![
                // Position among siblings
                child_idx as f32,
                // Total number of siblings
                tree.children(parent).len() as f32,
                // Is first child
                if child_idx == 0 { 1.0 } else { 0.0 },
                // Is last child
                if child_idx == tree.children(parent).len() - 1 { 1.0 } else { 0.0 },
            ];
            
            edge_features.push((parent, child, features));
        }
    }
    
    edge_features
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::TreeBuilder;

    #[test]
    fn test_feature_extraction() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1)
            .add_child(0, 2)
            .add_child(1, 3);
        let tree = builder.build().unwrap();
        
        let extractor = StandardFeatureExtractor::new();
        let features = extractor.extract_features(&tree);
        
        assert_eq!(features.len(), 4);
        assert!(features[0].is_root);
        assert_eq!(features[0].degree, 2); // Two children
        assert_eq!(features[0].subtree_size, 4);
        assert_eq!(features[0].num_leaves, 2);
        
        let array = extractor.features_to_array(&features);
        assert_eq!(array.shape(), &[4, 6]);
    }

    #[test]
    fn test_edge_features() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1)
            .add_child(0, 2);
        let tree = builder.build().unwrap();
        
        let edge_feats = extract_edge_features(&tree);
        assert_eq!(edge_feats.len(), 2);
        
        // First child
        assert_eq!(edge_feats[0].0, 0); // parent
        assert_eq!(edge_feats[0].1, 1); // child
        assert_eq!(edge_feats[0].2[0], 0.0); // position
        assert_eq!(edge_feats[0].2[2], 1.0); // is first
    }
} 