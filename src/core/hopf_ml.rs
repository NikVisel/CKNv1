//! Main HopfML integration structure

use crate::{
    algebra::{Tree, Forest, delta_k, CoProduct, Antipode},
    graph::{GraphData, tree_to_graph},
    Result,
};
use std::collections::HashMap;

/// Main structure integrating Hopf algebra with ML
pub struct HopfML {
    /// Cache of generated trees by size
    tree_cache: HashMap<usize, Vec<Tree>>,
    /// Maximum tree size to cache
    max_cache_size: usize,
}

impl HopfML {
    /// Create new HopfML instance
    pub fn new() -> Self {
        HopfML {
            tree_cache: HashMap::new(),
            max_cache_size: 10,
        }
    }

    /// Set maximum cache size
    pub fn with_max_cache_size(mut self, size: usize) -> Self {
        self.max_cache_size = size;
        self
    }

    /// Get all trees of given size (cached)
    pub fn get_trees(&mut self, size: usize) -> &Vec<Tree> {
        if size <= self.max_cache_size {
            self.tree_cache.entry(size).or_insert_with(|| delta_k(size))
        } else {
            // For large sizes, don't cache
            self.tree_cache.insert(size, delta_k(size));
            &self.tree_cache[&size]
        }
    }

    /// Generate dataset for predicting number of admissible cuts
    pub fn generate_cuts_dataset(&mut self, max_size: usize) -> Vec<(GraphData, f32)> {
        let mut dataset = Vec::new();
        
        for size in 1..=max_size {
            let trees = self.get_trees(size).clone();
            for tree in trees {
                let graph = tree_to_graph(&tree);
                let num_cuts = tree.admissible_cuts().len() as f32;
                dataset.push((graph, num_cuts));
            }
        }
        
        dataset
    }

    /// Generate dataset for antipode prediction
    pub fn generate_antipode_dataset(&mut self, max_size: usize) -> Vec<(GraphData, Vec<f32>)> {
        let mut dataset = Vec::new();
        
        for size in 1..=max_size {
            let trees = self.get_trees(size).clone();
            for tree in trees {
                let graph = tree_to_graph(&tree);
                
                // Encode antipode as feature vector
                let antipode = tree.antipode();
                let antipode_vec = encode_forest_as_vector(&antipode, max_size);
                
                dataset.push((graph, antipode_vec));
            }
        }
        
        dataset
    }

    /// Generate dataset for coproduct coefficient prediction
    pub fn generate_coproduct_dataset(&mut self, max_size: usize) -> Vec<(GraphData, Vec<f32>)> {
        let mut dataset = Vec::new();
        
        for size in 2..=max_size {
            let trees = self.get_trees(size).clone();
            for tree in trees {
                let graph = tree_to_graph(&tree);
                
                // Count coproduct coefficients by size
                let cop = tree.coproduct();
                let mut coeff_vec = vec![0.0; size];
                
                for ((forest, _trunk), &coeff) in cop.iter() {
                    let forest_size = forest.total_nodes();
                    if forest_size < size {
                        coeff_vec[forest_size] += coeff as f32;
                    }
                }
                
                dataset.push((graph, coeff_vec));
            }
        }
        
        dataset
    }

    /// Generate contrastive pairs for grafting
    pub fn generate_grafting_pairs(&mut self, max_size: usize) -> Vec<(GraphData, GraphData, bool)> {
        let mut pairs = Vec::new();
        
        for size in 1..max_size {
            let trees = self.get_trees(size).clone();
            
            for tree in trees {
                let grafted = tree.graft_all_leaves();
                
                // Positive pairs: tree and its grafted versions
                for g_tree in grafted.iter().take(2) {
                    let graph1 = tree_to_graph(&tree);
                    let graph2 = tree_to_graph(g_tree);
                    pairs.push((graph1, graph2, true));
                }
                
                // Negative pairs: tree and unrelated trees
                let other_trees = self.get_trees(size + 1);
                for other in other_trees.iter().take(2) {
                    if !grafted.contains(other) {
                        let graph1 = tree_to_graph(&tree);
                        let graph2 = tree_to_graph(other);
                        pairs.push((graph1, graph2, false));
                    }
                }
            }
        }
        
        pairs
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.tree_cache.clear();
    }
}

impl Default for HopfML {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode a forest as a fixed-size vector
fn encode_forest_as_vector(forest: &Forest, max_size: usize) -> Vec<f32> {
    let mut vec = vec![0.0; max_size + 1];
    
    for tree in forest.iter() {
        let size = tree.size();
        if size <= max_size {
            vec[size] += 1.0;
        }
    }
    
    vec
}

/// Statistics about tree generation
#[derive(Debug, Clone)]
pub struct TreeStatistics {
    pub size: usize,
    pub count: usize,
    pub avg_cuts: f32,
    pub max_cuts: usize,
    pub avg_height: f32,
    pub avg_leaves: f32,
}

impl HopfML {
    /// Compute statistics for trees of given size
    pub fn compute_statistics(&mut self, size: usize) -> TreeStatistics {
        let trees = self.get_trees(size).clone();
        let count = trees.len();
        
        if count == 0 {
            return TreeStatistics {
                size,
                count: 0,
                avg_cuts: 0.0,
                max_cuts: 0,
                avg_height: 0.0,
                avg_leaves: 0.0,
            };
        }
        
        let mut total_cuts = 0;
        let mut max_cuts = 0;
        let mut total_height = 0;
        let mut total_leaves = 0;
        
        for tree in &trees {
            let cuts = tree.admissible_cuts().len();
            total_cuts += cuts;
            max_cuts = max_cuts.max(cuts);
            
            let height = compute_tree_height(&tree);
            total_height += height;
            
            let leaves = count_tree_leaves(&tree);
            total_leaves += leaves;
        }
        
        TreeStatistics {
            size,
            count,
            avg_cuts: total_cuts as f32 / count as f32,
            max_cuts,
            avg_height: total_height as f32 / count as f32,
            avg_leaves: total_leaves as f32 / count as f32,
        }
    }
}

fn compute_tree_height(tree: &Tree) -> usize {
    fn height_recursive(tree: &Tree, node: usize) -> usize {
        let children = tree.children(node);
        if children.is_empty() {
            0
        } else {
            1 + children.iter()
                .map(|&child| height_recursive(tree, child))
                .max()
                .unwrap_or(0)
        }
    }
    
    height_recursive(tree, 0)
}

fn count_tree_leaves(tree: &Tree) -> usize {
    (0..tree.size())
        .filter(|&node| tree.children(node).is_empty())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hopf_ml_creation() {
        let mut hopf = HopfML::new();
        let trees = hopf.get_trees(3);
        assert_eq!(trees.len(), 2); // Two trees of size 3
    }

    #[test]
    fn test_cuts_dataset() {
        let mut hopf = HopfML::new();
        let dataset = hopf.generate_cuts_dataset(3);
        assert!(!dataset.is_empty());
        
        // Check that single node tree has 0 proper cuts
        let single_node_data = dataset.iter()
            .find(|(g, _)| g.num_nodes == 1)
            .unwrap();
        assert_eq!(single_node_data.1, 0.0);
    }

    #[test]
    fn test_statistics() {
        let mut hopf = HopfML::new();
        let stats = hopf.compute_statistics(4);
        
        assert_eq!(stats.size, 4);
        assert_eq!(stats.count, 4); // 4 trees of size 4
        assert!(stats.avg_cuts > 0.0);
    }
} 