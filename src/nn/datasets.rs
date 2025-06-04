//! Dataset utilities for Hopf algebra machine learning

use std::collections::HashMap;
use std::sync::Arc;
use ndarray::{Array1, Array2};
use num_rational::Rational64;
use crate::algebra::{Tree, Forest, HopfAlgebra};
use crate::graph::{tree_to_graph, GraphData};

/// Generate all trees up to a given size
fn generate_trees_up_to_size(max_size: usize) -> Vec<Tree> {
    let mut all_trees = vec![Tree::new()]; // Start with single node
    let mut result = vec![Tree::new()];
    
    for size in 2..=max_size {
        let mut new_trees = Vec::new();
        
        // Generate trees of size `size` by grafting
        for base_tree in &all_trees {
            if base_tree.size() == size - 1 {
                let grafted = base_tree.graft_all_leaves();
                for tree in grafted {
                    if tree.size() == size {
                        new_trees.push(tree);
                    }
                }
            }
        }
        
        // Deduplicate by canonical form
        let mut unique_trees = Vec::new();
        let mut seen_forms = std::collections::HashSet::new();
        
        for tree in new_trees {
            let form = format!("{:?}", tree); // Use debug format as canonical form
            if seen_forms.insert(form) {
                unique_trees.push(tree);
            }
        }
        
        result.extend(unique_trees.clone());
        all_trees.extend(unique_trees);
    }
    
    result
}

/// Dataset for predicting admissible cuts
pub struct AdmissibleCutsDataset {
    /// Trees and their admissible cuts
    samples: Vec<(Tree, Vec<Vec<usize>>)>,
}

impl AdmissibleCutsDataset {
    /// Create a new dataset
    pub fn new(max_tree_size: usize) -> Self {
        let trees = generate_trees_up_to_size(max_tree_size);
        let hopf = HopfAlgebra::new(100);
        
        let samples = trees.into_iter()
            .map(|tree| {
                let cuts = hopf.admissible_cuts(&tree);
                (tree, cuts)
            })
            .collect();
            
        AdmissibleCutsDataset { samples }
    }
    
    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    
    /// Get a sample
    pub fn get(&self, idx: usize) -> Option<&(Tree, Vec<Vec<usize>>)> {
        self.samples.get(idx)
    }
    
    /// Convert to graph representation
    pub fn to_graph(&self, idx: usize) -> Option<(GraphData, Vec<Vec<usize>>)> {
        let (tree, cuts) = self.get(idx)?;
        let graph = tree_to_graph(tree);
        Some((graph, cuts.clone()))
    }
}

/// Dataset for learning antipode values
pub struct AntipodeDataset {
    /// (input_tree, antipode_tree) pairs
    samples: Vec<(Tree, Tree)>,
}

impl AntipodeDataset {
    /// Create a new dataset
    pub fn new(max_tree_size: usize) -> Self {
        let trees = generate_trees_up_to_size(max_tree_size);
        let hopf = Arc::new(HopfAlgebra::new(100));
        
        let samples = trees.into_iter()
            .map(|tree| {
                let antipode = hopf.antipode(&tree);
                (tree, antipode)
            })
            .collect();
            
        AntipodeDataset { samples }
    }
    
    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    
    /// Get a sample
    pub fn get(&self, idx: usize) -> Option<&(Tree, Tree)> {
        self.samples.get(idx)
    }
}

/// Dataset for grafting prediction
pub struct GraftingDataset {
    /// (base_tree, graft_position, result_tree) tuples
    samples: Vec<(Tree, usize, Tree)>,
}

impl GraftingDataset {
    /// Create a new dataset
    pub fn new(max_tree_size: usize) -> Self {
        let trees = generate_trees_up_to_size(max_tree_size.saturating_sub(1));
        let mut samples = Vec::new();
        
        for tree in trees {
            let grafted = tree.graft_all_leaves();
            for (pos, new_tree) in grafted.into_iter().enumerate() {
                if new_tree.size() <= max_tree_size {
                    samples.push((tree.clone(), pos, new_tree));
                }
            }
        }
        
        GraftingDataset { samples }
    }
    
    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    
    /// Get a sample
    pub fn get(&self, idx: usize) -> Option<&(Tree, usize, Tree)> {
        self.samples.get(idx)
    }
}

/// Dataset for learning coproduct coefficient prediction
pub struct CoefficientDataset {
    /// (tree, cut, coefficient) tuples
    samples: Vec<(Tree, Vec<usize>, f64)>,
    /// Rational coefficients from exact computation
    exact_coefficients: HashMap<(String, String), Rational64>,
}

impl CoefficientDataset {
    /// Create a new coefficient dataset
    pub fn new(hopf: Arc<HopfAlgebra>, max_tree_size: usize) -> crate::Result<Self> {
        let mut samples = Vec::new();
        let mut exact_coefficients = HashMap::new();
        
        // Generate trees up to max_size
        let trees = generate_trees_up_to_size(max_tree_size);
        
        for tree in trees {
            // Get all admissible cuts
            let cuts = hopf.admissible_cuts(&tree);
            
            // For each cut, compute the exact coefficient
            for cut in cuts {
                let (left, right) = hopf.apply_cut(&tree, &cut)?;
                
                // Get coefficient from coproduct
                let coproduct = hopf.coproduct(&tree);
                let coeff = coproduct.coefficient(&left, &right).unwrap_or(Rational64::zero());
                
                // Store exact rational coefficient
                let key = (format!("{:?}", left), format!("{:?}", right));
                exact_coefficients.insert(key, coeff);
                
                // Convert to float for learning
                let coeff_f64 = *coeff.numer() as f64 / *coeff.denom() as f64;
                
                samples.push((tree.clone(), cut.clone(), coeff_f64));
            }
        }
        
        Ok(CoefficientDataset {
            samples,
            exact_coefficients,
        })
    }
    
    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    
    /// Get a sample by index
    pub fn get(&self, idx: usize) -> Option<&(Tree, Vec<usize>, f64)> {
        self.samples.get(idx)
    }
    
    /// Create features for coefficient prediction
    /// Returns (tree_features, cut_features, coefficient)
    pub fn featurize(&self, idx: usize) -> Option<(Array1<f32>, Array1<f32>, f32)> {
        let (tree, cut, coeff) = self.get(idx)?;
        
        // Tree features
        let mut tree_features = Array1::zeros(10);
        tree_features[0] = tree.size() as f32;
        tree_features[1] = tree.max_depth() as f32;
        tree_features[2] = tree.leaf_count() as f32;
        tree_features[3] = tree.hopf_degree() as f32;
        
        // Cut-specific features
        tree_features[4] = cut.len() as f32;
        tree_features[5] = cut.iter().map(|&n| tree.subtree_size(n)).sum::<usize>() as f32;
        tree_features[6] = cut.iter().map(|&n| tree.node_depth(n)).max().unwrap_or(0) as f32;
        
        // Check if cut is "balanced" (similar subtree sizes)
        if cut.len() > 1 {
            let sizes: Vec<usize> = cut.iter().map(|&n| tree.subtree_size(n)).collect();
            let mean_size = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;
            let variance = sizes.iter()
                .map(|&s| (s as f32 - mean_size).powi(2))
                .sum::<f32>() / sizes.len() as f32;
            tree_features[7] = variance.sqrt(); // Standard deviation
        }
        
        // Cut encoding (one-hot or positional)
        let mut cut_features = Array1::zeros(tree.size());
        for &node in cut {
            cut_features[node] = 1.0;
        }
        
        Some((tree_features, cut_features, *coeff as f32))
    }
    
    /// Split into train/test sets
    pub fn train_test_split(&self, test_fraction: f32) -> (Vec<usize>, Vec<usize>) {
        let n = self.len();
        let n_test = (n as f32 * test_fraction) as usize;
        
        let mut indices: Vec<usize> = (0..n).collect();
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        
        let test_indices = indices[..n_test].to_vec();
        let train_indices = indices[n_test..].to_vec();
        
        (train_indices, test_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_datasets_creation() {
        let cuts_dataset = AdmissibleCutsDataset::new(3);
        assert!(!cuts_dataset.is_empty());
        
        let antipode_dataset = AntipodeDataset::new(3);
        assert!(!antipode_dataset.is_empty());
        
        let grafting_dataset = GraftingDataset::new(3);
        assert!(!grafting_dataset.is_empty());
    }
    
    #[test]
    fn test_coefficient_dataset() {
        let hopf = Arc::new(HopfAlgebra::new(100));
        let dataset = CoefficientDataset::new(hopf, 4).unwrap();
        
        assert!(!dataset.is_empty());
        
        // Check that we can featurize samples
        if let Some((tree_feat, cut_feat, coeff)) = dataset.featurize(0) {
            assert!(tree_feat.len() > 0);
            assert!(cut_feat.len() > 0);
            assert!(coeff.is_finite());
        }
    }
} 