//! Convert trees to graph representations

use crate::algebra::Tree;
use super::{GraphData, EdgeIndex, FeatureExtractor, StandardFeatureExtractor};
use ndarray::{Array1, Array2};

/// Configuration for tree to graph conversion
#[derive(Clone, Debug)]
pub struct ConversionConfig {
    /// Make graph undirected
    pub undirected: bool,
    /// Include self-loops
    pub self_loops: bool,
    /// Include edge features
    pub edge_features: bool,
    /// Include graph-level features
    pub graph_features: bool,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        ConversionConfig {
            undirected: true,
            self_loops: true,
            edge_features: false,
            graph_features: false,
        }
    }
}

/// Convert a tree to graph data using default features
pub fn tree_to_graph(tree: &Tree) -> GraphData {
    let config = ConversionConfig::default();
    let extractor = StandardFeatureExtractor::new();
    tree_to_graph_with_config(tree, &config, &extractor)
}

/// Convert a tree to graph data with custom configuration
pub fn tree_to_graph_with_config<E: FeatureExtractor>(
    tree: &Tree,
    config: &ConversionConfig,
    extractor: &E,
) -> GraphData {
    let n = tree.size();
    
    // Extract node features
    let node_features = extractor.extract_features(tree);
    let node_feature_array = extractor.features_to_array(&node_features);
    
    // Build edge index
    let mut edges = Vec::new();
    for parent in 0..n {
        for &child in tree.children(parent) {
            edges.push((parent, child));
            if config.undirected {
                edges.push((child, parent));
            }
        }
    }
    
    if config.self_loops {
        for i in 0..n {
            edges.push((i, i));
        }
    }
    
    let edge_index = EdgeIndex::from_edges(edges);
    
    // Create basic graph data
    let mut graph = GraphData::new(n, edge_index, node_feature_array);
    
    // Add edge features if requested
    if config.edge_features {
        let edge_feats = super::features::extract_edge_features(tree);
        let num_edges = graph.edge_index.num_edges();
        let edge_dim = if edge_feats.is_empty() { 4 } else { edge_feats[0].2.len() };
        
        let mut edge_array = Array2::zeros((num_edges, edge_dim));
        let mut edge_map = std::collections::HashMap::new();
        
        // Map edges to indices
        for (idx, (s, d)) in graph.edge_index.src.iter()
            .zip(&graph.edge_index.dst)
            .enumerate() 
        {
            edge_map.insert((*s, *d), idx);
        }
        
        // Fill edge features
        for (parent, child, features) in edge_feats {
            if let Some(&idx) = edge_map.get(&(parent, child)) {
                for (j, &val) in features.iter().enumerate() {
                    edge_array[[idx, j]] = val;
                }
            }
            // Handle reverse edge if undirected
            if config.undirected {
                if let Some(&idx) = edge_map.get(&(child, parent)) {
                    // Reverse edge features (swap first/last child indicators)
                    let mut rev_features = features.clone();
                    let tmp = rev_features[2];
                    rev_features[2] = rev_features[3];
                    rev_features[3] = tmp;
                    for (j, &val) in rev_features.iter().enumerate() {
                        edge_array[[idx, j]] = val;
                    }
                }
            }
        }
        
        graph = graph.with_edge_features(edge_array);
    }
    
    // Add graph features if requested
    if config.graph_features {
        let graph_feats = compute_graph_features(tree);
        graph = graph.with_graph_features(graph_feats);
    }
    
    graph
}

/// Compute graph-level features
fn compute_graph_features(tree: &Tree) -> Array1<f32> {
    use crate::algebra::CoProduct;
    
    let n = tree.size() as f32;
    let num_edges = tree.children(0).len() as f32;
    let num_leaves = count_leaves(tree) as f32;
    
    // Tree statistics
    let avg_degree = if n > 0.0 { 2.0 * num_edges / n } else { 0.0 };
    let leaf_ratio = if n > 0.0 { num_leaves / n } else { 0.0 };
    
    // Hopf-algebraic features
    let num_cuts = tree.admissible_cuts().len() as f32;
    let cuts_per_node = if n > 0.0 { num_cuts / n } else { 0.0 };
    
    Array1::from(vec![
        n,
        num_edges,
        num_leaves,
        avg_degree,
        leaf_ratio,
        num_cuts,
        cuts_per_node,
    ])
}

fn count_leaves(tree: &Tree) -> usize {
    (0..tree.size())
        .filter(|&node| tree.children(node).is_empty())
        .count()
}

/// High-level wrapper for tree graph conversion
pub struct TreeGraph {
    tree: Tree,
    config: ConversionConfig,
}

impl TreeGraph {
    /// Create new tree graph
    pub fn new(tree: Tree) -> Self {
        TreeGraph {
            tree,
            config: ConversionConfig::default(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: ConversionConfig) -> Self {
        self.config = config;
        self
    }

    /// Convert to graph data
    pub fn to_graph_data<E: FeatureExtractor>(&self, extractor: &E) -> GraphData {
        tree_to_graph_with_config(&self.tree, &self.config, extractor)
    }

    /// Get the underlying tree
    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    /// Create a batch of graph data from multiple trees
    pub fn batch_from_trees<E: FeatureExtractor>(
        trees: Vec<Tree>,
        config: &ConversionConfig,
        extractor: &E,
    ) -> Vec<GraphData> {
        trees.into_iter()
            .map(|tree| tree_to_graph_with_config(&tree, config, extractor))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::TreeBuilder;

    #[test]
    fn test_tree_to_graph() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1)
            .add_child(0, 2);
        let tree = builder.build().unwrap();
        
        let graph = tree_to_graph(&tree);
        
        assert_eq!(graph.num_nodes, 3);
        assert_eq!(graph.feature_dim(), 6); // Standard features
        
        // Check edges (undirected + self-loops)
        let expected_edges = 2 * 2 + 3; // 2 edges * 2 directions + 3 self-loops
        assert_eq!(graph.edge_index.num_edges(), expected_edges);
    }

    #[test]
    fn test_custom_config() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1);
        let tree = builder.build().unwrap();
        
        let config = ConversionConfig {
            undirected: false,
            self_loops: false,
            edge_features: true,
            graph_features: true,
        };
        
        let extractor = StandardFeatureExtractor::new();
        let graph = tree_to_graph_with_config(&tree, &config, &extractor);
        
        assert_eq!(graph.edge_index.num_edges(), 1); // Only parent->child
        assert!(graph.edge_features.is_some());
        assert!(graph.graph_features.is_some());
    }
} 