//! Graph data structures for neural networks

use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};

/// Edge index representation for sparse adjacency
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeIndex {
    /// Source nodes for each edge
    pub src: Vec<usize>,
    /// Destination nodes for each edge
    pub dst: Vec<usize>,
}

impl EdgeIndex {
    /// Create from edge list
    pub fn from_edges(edges: Vec<(usize, usize)>) -> Self {
        let (src, dst): (Vec<_>, Vec<_>) = edges.into_iter().unzip();
        EdgeIndex { src, dst }
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.src.len()
    }

    /// Convert to undirected (add reverse edges)
    pub fn to_undirected(&self) -> Self {
        let mut src = self.src.clone();
        let mut dst = self.dst.clone();
        
        // Add reverse edges
        src.extend(&self.dst);
        dst.extend(&self.src);
        
        EdgeIndex { src, dst }
    }

    /// Get adjacency list representation
    pub fn to_adjacency_list(&self, num_nodes: usize) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); num_nodes];
        
        for (s, d) in self.src.iter().zip(&self.dst) {
            adj[*s].push(*d);
        }
        
        adj
    }
}

/// Graph data for neural network processing
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphData {
    /// Number of nodes
    pub num_nodes: usize,
    
    /// Edge connectivity
    pub edge_index: EdgeIndex,
    
    /// Node features matrix [num_nodes, feature_dim]
    pub node_features: Array2<f32>,
    
    /// Optional edge features [num_edges, edge_feature_dim]
    pub edge_features: Option<Array2<f32>>,
    
    /// Optional graph-level features
    pub graph_features: Option<Array1<f32>>,
}

impl GraphData {
    /// Create new graph data
    pub fn new(
        num_nodes: usize,
        edge_index: EdgeIndex,
        node_features: Array2<f32>,
    ) -> Self {
        assert_eq!(node_features.shape()[0], num_nodes);
        
        GraphData {
            num_nodes,
            edge_index,
            node_features,
            edge_features: None,
            graph_features: None,
        }
    }

    /// Set edge features
    pub fn with_edge_features(mut self, features: Array2<f32>) -> Self {
        assert_eq!(features.shape()[0], self.edge_index.num_edges());
        self.edge_features = Some(features);
        self
    }

    /// Set graph features
    pub fn with_graph_features(mut self, features: Array1<f32>) -> Self {
        self.graph_features = Some(features);
        self
    }

    /// Get feature dimension
    pub fn feature_dim(&self) -> usize {
        self.node_features.shape()[1]
    }

    /// Build adjacency matrix (dense)
    pub fn adjacency_matrix(&self) -> Array2<f32> {
        let n = self.num_nodes;
        let mut adj = Array2::zeros((n, n));
        
        for (s, d) in self.edge_index.src.iter().zip(&self.edge_index.dst) {
            adj[[*s, *d]] = 1.0;
        }
        
        adj
    }

    /// Build degree matrix
    pub fn degree_matrix(&self) -> Array1<f32> {
        let mut degrees = Array1::zeros(self.num_nodes);
        
        for s in &self.edge_index.src {
            degrees[*s] += 1.0;
        }
        
        degrees
    }

    /// Create adjacency mask for attention mechanisms
    pub fn attention_mask(&self, include_self: bool) -> Array2<bool> {
        let n = self.num_nodes;
        let mut mask = Array2::from_elem((n, n), false);
        
        // Add edges
        for (s, d) in self.edge_index.src.iter().zip(&self.edge_index.dst) {
            mask[[*s, *d]] = true;
        }
        
        // Add self-loops if requested
        if include_self {
            for i in 0..n {
                mask[[i, i]] = true;
            }
        }
        
        mask
    }

    /// Compute distance matrix (shortest paths)
    pub fn distance_matrix(&self) -> Array2<i32> {
        let n = self.num_nodes;
        let adj_list = self.edge_index.to_adjacency_list(n);
        let mut dist = Array2::from_elem((n, n), i32::MAX / 2);
        
        // Initialize distances
        for i in 0..n {
            dist[[i, i]] = 0;
        }
        
        for (s, d) in self.edge_index.src.iter().zip(&self.edge_index.dst) {
            dist[[*s, *d]] = 1;
        }
        
        // Floyd-Warshall
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let through_k = dist[[i, k]].saturating_add(dist[[k, j]]);
                    if through_k < dist[[i, j]] {
                        dist[[i, j]] = through_k;
                    }
                }
            }
        }
        
        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_edge_index() {
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let ei = EdgeIndex::from_edges(edges);
        
        assert_eq!(ei.num_edges(), 3);
        
        let undirected = ei.to_undirected();
        assert_eq!(undirected.num_edges(), 6);
    }

    #[test]
    fn test_graph_data() {
        let ei = EdgeIndex::from_edges(vec![(0, 1), (1, 2)]);
        let features = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        
        let graph = GraphData::new(3, ei, features);
        assert_eq!(graph.num_nodes, 3);
        assert_eq!(graph.feature_dim(), 2);
        
        let adj = graph.adjacency_matrix();
        assert_eq!(adj[[0, 1]], 1.0);
        assert_eq!(adj[[0, 2]], 0.0);
    }
} 