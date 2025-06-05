//! Tree autoencoder architectures for learning tree representations

use std::sync::Arc;
use ndarray::{Array1, Array2, Array3};
use crate::algebra::{Tree, Forest, HopfAlgebra, TreeBuilder};
use crate::graph::{GraphData, tree_to_graph};

/// Configuration for tree autoencoder
#[derive(Debug, Clone)]
pub struct AutoencoderConfig {
    /// Dimension of latent space
    pub latent_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Use grafting-aware architecture
    pub grafting_aware: bool,
    /// Use Hopf-algebraic constraints
    pub hopf_constrained: bool,
    /// Dropout rate
    pub dropout: f32,
}

impl Default for AutoencoderConfig {
    fn default() -> Self {
        AutoencoderConfig {
            latent_dim: 128,
            hidden_dims: vec![256, 512, 256],
            grafting_aware: true,
            hopf_constrained: true,
            dropout: 0.1,
        }
    }
}

/// Grafted-tree autoencoder
pub struct GraftedTreeAutoencoder {
    config: AutoencoderConfig,
    hopf: Arc<HopfAlgebra>,
}

impl GraftedTreeAutoencoder {
    /// Create a new autoencoder
    pub fn new(config: AutoencoderConfig, hopf: Arc<HopfAlgebra>) -> Self {
        GraftedTreeAutoencoder { config, hopf }
    }
    
    /// Encode a tree to latent representation
    pub fn encode(&self, tree: &Tree) -> Array1<f32> {
        // Convert tree to graph
        let graph = tree_to_graph(tree);
        
        // Extract features
        let node_features = self.extract_node_features(tree, &graph);
        let structural_code = self.encode_structure(tree);
        let grafting_code = if self.config.grafting_aware {
            self.encode_grafting_potential(tree)
        } else {
            Array1::zeros(32)
        };
        
        // Combine all features
        let mut combined = Array1::zeros(self.config.latent_dim);
        let feat_dim = node_features.ncols();
        
        // Aggregate node features (mean pooling)
        for i in 0..feat_dim.min(self.config.latent_dim / 3) {
            combined[i] = node_features.column(i).mean().unwrap_or(0.0);
        }
        
        // Add structural encoding
        let struct_start = self.config.latent_dim / 3;
        let struct_end = 2 * self.config.latent_dim / 3;
        for (i, &val) in structural_code.iter().enumerate() {
            if struct_start + i < struct_end {
                combined[struct_start + i] = val;
            }
        }
        
        // Add grafting encoding
        let graft_start = 2 * self.config.latent_dim / 3;
        for (i, &val) in grafting_code.iter().enumerate() {
            if graft_start + i < self.config.latent_dim {
                combined[graft_start + i] = val;
            }
        }
        
        combined
    }
    
    /// Decode from latent representation to tree
    pub fn decode(&self, latent: &Array1<f32>) -> Tree {
        // This is a placeholder - in practice would use neural decoder
        // For now, decode based on latent vector properties
        
        let size_estimate = (latent[0].abs() * 10.0) as usize + 1;
        let size = size_estimate.min(10).max(1);
        
        // Generate a tree based on latent encoding
        self.generate_tree_from_encoding(latent, size)
    }
    
    /// Extract node features for encoding
    fn extract_node_features(&self, tree: &Tree, graph: &GraphData) -> Array2<f32> {
        let n_nodes = tree.size();
        let n_features = 16;
        let mut features = Array2::zeros((n_nodes, n_features));
        
        for node in 0..n_nodes {
            features[(node, 0)] = tree.node_depth(node) as f32;
            features[(node, 1)] = tree.subtree_size(node) as f32;
            features[(node, 2)] = tree.node_degree(node) as f32;
            features[(node, 3)] = if tree.node_degree(node) == 0 { 1.0 } else { 0.0 };
            
            // Hopf-algebraic features
            if self.config.hopf_constrained {
                let subtree = self.extract_subtree(tree, node);
                let antipode = self.hopf.tree_antipode(&subtree);
                features[(node, 4)] = antipode.size() as f32;
                features[(node, 5)] = antipode.max_depth() as f32;
                
                // Coproduct features
                let cuts = self.hopf.admissible_cuts(&subtree);
                features[(node, 6)] = cuts.len() as f32;
            }
            
            // Path-based features
            let path = tree.path_to_root(node);
            features[(node, 7)] = path.len() as f32;
            features[(node, 8)] = path.iter()
                .map(|&n| tree.node_degree(n) as f32)
                .sum::<f32>() / path.len().max(1) as f32;
        }
        
        features
    }
    
    /// Encode tree structure
    fn encode_structure(&self, tree: &Tree) -> Array1<f32> {
        let mut encoding = Array1::zeros(64);
        
        // Basic structural properties
        encoding[0] = tree.size() as f32;
        encoding[1] = tree.max_depth() as f32;
        encoding[2] = tree.leaf_count() as f32;
        encoding[3] = tree.hopf_degree() as f32;
        
        // Depth distribution
        let max_depth = tree.max_depth();
        for d in 0..=max_depth.min(10) {
            let nodes_at_d = tree.nodes_at_depth(d);
            encoding[4 + d] = nodes_at_d.len() as f32;
        }
        
        // Branching factor statistics
        let mut branching_factors = Vec::new();
        for node in 0..tree.size() {
            let degree = tree.node_degree(node);
            if degree > 0 {
                branching_factors.push(degree as f32);
            }
        }
        
        if !branching_factors.is_empty() {
            encoding[15] = branching_factors.iter().sum::<f32>() / branching_factors.len() as f32;
            encoding[16] = branching_factors
                .iter()
                .cloned()
                .fold(f32::MIN, f32::max);
            encoding[17] = branching_factors
                .iter()
                .cloned()
                .fold(f32::MAX, f32::min);
        }
        
        encoding
    }
    
    /// Encode grafting potential
    fn encode_grafting_potential(&self, tree: &Tree) -> Array1<f32> {
        let mut encoding = Array1::zeros(32);
        
        // Analyze grafting positions
        let grafted_trees = tree.graft_all_leaves();
        encoding[0] = grafted_trees.len() as f32;
        
        // Analyze how grafting changes tree properties
        let mut size_changes = Vec::new();
        let mut depth_changes = Vec::new();
        
        for grafted in &grafted_trees {
            size_changes.push((grafted.size() - tree.size()) as f32);
            depth_changes.push((grafted.max_depth() as i32 - tree.max_depth() as i32) as f32);
        }
        
        if !size_changes.is_empty() {
            encoding[1] = size_changes.iter().sum::<f32>() / size_changes.len() as f32;
            encoding[2] = depth_changes.iter().sum::<f32>() / depth_changes.len() as f32;
        }
        
        // Identify optimal grafting positions
        for (i, grafted) in grafted_trees.iter().enumerate().take(10) {
            // Score based on balance and growth
            let score = self.score_grafted_tree(tree, grafted);
            encoding[3 + i] = score;
        }
        
        encoding
    }
    
    /// Score a grafted tree for quality
    fn score_grafted_tree(&self, original: &Tree, grafted: &Tree) -> f32 {
        let mut score = 0.0;
        
        // Prefer balanced growth
        let balance = self.compute_balance(grafted);
        score += balance;
        
        // Prefer maintaining depth constraints
        let depth_ratio = grafted.max_depth() as f32 / original.max_depth().max(1) as f32;
        score += 1.0 / (1.0 + (depth_ratio - 1.5).abs());
        
        // Hopf-algebraic score
        if self.config.hopf_constrained {
            let cuts = self.hopf.admissible_cuts(grafted);
            score += (cuts.len() as f32).ln();
        }
        
        score
    }
    
    /// Compute tree balance
    fn compute_balance(&self, tree: &Tree) -> f32 {
        let mut balance = 0.0;
        let mut count = 0;
        
        for node in 0..tree.size() {
            let children = tree.children(node);
            if children.len() >= 2 {
                let sizes: Vec<f32> = children.iter()
                    .map(|&c| tree.subtree_size(c) as f32)
                    .collect();
                let mean = sizes.iter().sum::<f32>() / sizes.len() as f32;
                let variance = sizes.iter()
                    .map(|&s| (s - mean).powi(2))
                    .sum::<f32>() / sizes.len() as f32;
                balance += 1.0 / (1.0 + variance.sqrt());
                count += 1;
            }
        }
        
        if count > 0 {
            balance / count as f32
        } else {
            1.0
        }
    }
    
    /// Extract subtree rooted at node
    fn extract_subtree(&self, tree: &Tree, root: usize) -> Tree {
        use std::collections::VecDeque;
        use std::collections::HashMap;

        // Map original node indices to new ones
        let mut map = HashMap::new();
        map.insert(root, 0usize);

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new()];
        let mut queue = VecDeque::new();
        queue.push_back(root);

        while let Some(node) = queue.pop_front() {
            let parent_new = map[&node];
            for &child in tree.children(node) {
                // Assign new index for child
                let new_idx = adjacency.len();
                adjacency.push(Vec::new());
                adjacency[parent_new].push(new_idx);
                map.insert(child, new_idx);
                queue.push_back(child);
            }
        }

        Tree::from_adjacency(adjacency).unwrap_or_else(|_| Tree::new())
    }
    
    /// Generate tree from encoding
    fn generate_tree_from_encoding(&self, latent: &Array1<f32>, target_size: usize) -> Tree {
        use crate::algebra::TreeBuilder;
        
        // Start with root
        let mut builder = TreeBuilder::new();
        let mut current_size = 1;
        let mut node_counter = 1;
        
        // Use latent vector to guide generation
        let mut rng = rand::thread_rng();
        use rand::Rng;
        
        while current_size < target_size {
            // Decide which node to extend based on latent encoding
            let extend_node = if current_size == 1 {
                0
            } else {
                let idx = (latent[current_size % latent.len()].abs() * current_size as f32) as usize;
                idx % current_size
            };
            
            // Decide how many children based on latent
            let n_children = ((latent[(current_size + 1) % latent.len()].abs() * 3.0) as usize + 1).min(3);
            
            for _ in 0..n_children.min(target_size - current_size) {
                builder.add_child(extend_node, node_counter);
                node_counter += 1;
                current_size += 1;
            }
            
            if current_size >= target_size {
                break;
            }
        }
        
        builder.build().unwrap_or_else(|_| Tree::new())
    }
}

/// Masked tree reconstruction task
pub struct MaskedTreeReconstruction {
    /// Masking probability
    mask_prob: f32,
    /// Use structural masking (mask entire subtrees)
    structural_masking: bool,
}

impl MaskedTreeReconstruction {
    /// Create new masked reconstruction task
    pub fn new(mask_prob: f32, structural_masking: bool) -> Self {
        MaskedTreeReconstruction {
            mask_prob,
            structural_masking,
        }
    }
    
    /// Apply masking to a tree
    pub fn mask_tree(&self, tree: &Tree) -> (Tree, Vec<usize>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut masked_nodes = Vec::new();
        
        if self.structural_masking {
            // Mask entire subtrees
            for node in 0..tree.size() {
                if rng.gen::<f32>() < self.mask_prob {
                    // Mask this subtree
                    self.collect_subtree_nodes(tree, node, &mut masked_nodes);
                }
            }
        } else {
            // Random node masking
            for node in 0..tree.size() {
                if rng.gen::<f32>() < self.mask_prob {
                    masked_nodes.push(node);
                }
            }
        }
        
        use std::collections::HashSet;

        let mask_set: HashSet<usize> = masked_nodes.iter().cloned().collect();

        // Build new adjacency excluding masked nodes and subtrees
        let mut adjacency = vec![Vec::new(); tree.size()];
        for parent in 0..tree.size() {
            if mask_set.contains(&parent) {
                continue;
            }
            for &child in tree.children(parent) {
                if !mask_set.contains(&child) {
                    adjacency[parent].push(child);
                }
            }
        }

        let masked_tree = Tree::from_adjacency(adjacency).unwrap_or_else(|_| tree.clone());
        (masked_tree, masked_nodes)
    }
    
    /// Collect all nodes in a subtree
    fn collect_subtree_nodes(&self, tree: &Tree, root: usize, nodes: &mut Vec<usize>) {
        nodes.push(root);
        for &child in tree.children(root) {
            self.collect_subtree_nodes(tree, child, nodes);
        }
    }
    
    /// Compute reconstruction loss
    pub fn reconstruction_loss(
        &self,
        original: &Tree,
        reconstructed: &Tree,
        masked_nodes: &[usize],
    ) -> f32 {
        // Simple loss based on structure difference
        let size_diff = (original.size() as f32 - reconstructed.size() as f32).abs();
        let depth_diff = (original.max_depth() as f32 - reconstructed.max_depth() as f32).abs();
        
        // Focus on masked regions
        let masked_penalty = masked_nodes.len() as f32 * 0.5;
        
        size_diff + depth_diff + masked_penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::TreeBuilder;
    
    #[test]
    fn test_autoencoder() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1)
            .add_child(0, 2)
            .add_child(1, 3);
        let tree = builder.build().unwrap();
            
        let hopf = Arc::new(HopfAlgebra::new(100));
        let config = AutoencoderConfig::default();
        let autoencoder = GraftedTreeAutoencoder::new(config, hopf);
        
        let encoding = autoencoder.encode(&tree);
        assert_eq!(encoding.len(), 128);
        
        let decoded = autoencoder.decode(&encoding);
        assert!(decoded.size() > 0);
    }
    
    #[test]
    fn test_masked_reconstruction() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1)
            .add_child(0, 2);
        let tree = builder.build().unwrap();
            
        let masker = MaskedTreeReconstruction::new(0.3, false);
        let (masked_tree, masked_nodes) = masker.mask_tree(&tree);
        
        assert!(masked_nodes.len() <= tree.size());
    }
} 