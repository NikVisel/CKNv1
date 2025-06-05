//! Renormalization-inspired applications

use crate::{
    algebra::{Tree, Forest, CoProduct, Antipode},
    graph::tree_to_graph,
    core::HopfML,
    Result,
};

/// Simulate renormalization of a Feynman-like graph
#[derive(Debug, Clone)]
pub struct RenormalizationSimulator {
    /// Coupling constant
    coupling: f32,
    /// Regularization scale
    scale: f32,
}

impl RenormalizationSimulator {
    /// Create new simulator
    pub fn new(coupling: f32, scale: f32) -> Self {
        RenormalizationSimulator { coupling, scale }
    }

    /// Compute "divergence degree" of a tree (toy model)
    pub fn divergence_degree(&self, tree: &Tree) -> i32 {
        // Toy model: degree = 2 * loops - 4
        // For trees: loops = edges - nodes + 1 = 0
        // So we use: degree = size - 2 * height
        let height = self.compute_height(tree);
        tree.size() as i32 - 2 * height as i32
    }

    fn compute_height(&self, tree: &Tree) -> usize {
        fn height_rec(tree: &Tree, node: usize) -> usize {
            let children = tree.children(node);
            if children.is_empty() {
                0
            } else {
                1 + children.iter()
                    .map(|&c| height_rec(tree, c))
                    .max()
                    .unwrap_or(0)
            }
        }
        height_rec(tree, 0)
    }

    /// Compute "regularized value" using forest formula
    pub fn regularized_value(&self, tree: &Tree) -> f32 {
        let div_degree = self.divergence_degree(tree);
        
        if div_degree <= 0 {
            // Convergent - no regularization needed
            self.bare_value(tree)
        } else {
            // Apply forest formula
            let mut value = self.bare_value(tree);
            
            // Subtract divergent subgraphs
            for cut in tree.admissible_cuts() {
                if !cut.pruned_forest.is_empty() {
                    let forest_val = self.forest_value(&cut.pruned_forest);
                    let trunk_val = self.regularized_value(&cut.trunk);
                    value -= forest_val * trunk_val;
                }
            }
            
            value
        }
    }

    /// Compute "bare value" (before regularization)
    fn bare_value(&self, tree: &Tree) -> f32 {
        // Toy model: coupling^(#vertices) * scale^(-divergence)
        let n = tree.size() as f32;
        let div = self.divergence_degree(tree) as f32;
        
        self.coupling.powf(n) * self.scale.powf(-div.max(0.0))
    }

    /// Compute value of a forest
    fn forest_value(&self, forest: &Forest) -> f32 {
        forest.iter()
            .map(|tree| self.regularized_value(tree))
            .product()
    }

    /// Compute counterterm using antipode
    pub fn counterterm(&self, tree: &Tree) -> f32 {
        let antipode = tree.antipode();
        -self.forest_value(&antipode)
    }
}

/// Analyze renormalization structure of trees
pub struct RenormalizationAnalyzer;

impl RenormalizationAnalyzer {
    /// Find minimal subtraction forest
    pub fn minimal_forest(tree: &Tree) -> Forest {
        // Find all divergent subgraphs
        let divergent_cuts: Vec<_> = tree.admissible_cuts()
            .into_iter()
            .filter(|cut| {
                // Check if pruned trees are "divergent" (toy criterion)
                cut.pruned_forest.iter().any(|t| t.size() > 2)
            })
            .collect();

        if divergent_cuts.is_empty() {
            return Forest::empty();
        }

        // Find minimal set (greedy algorithm)
        let mut selected = Forest::empty();
        let mut covered = std::collections::HashSet::new();

        for cut in divergent_cuts {
            let mut new_coverage = false;
            for tree in cut.pruned_forest.iter() {
                if !covered.contains(&tree_id(tree)) {
                    new_coverage = true;
                    covered.insert(tree_id(tree));
                }
            }

            if new_coverage {
                selected = selected.multiply(&cut.pruned_forest);
            }
        }

        selected
    }

    /// Compute renormalization group flow
    pub fn rg_flow(tree: &Tree, steps: usize) -> Vec<f32> {
        let mut trajectory = Vec::with_capacity(steps);
        let mut sim = RenormalizationSimulator::new(0.1, 1.0);

        for step in 0..steps {
            let scale = (step as f32 + 1.0).ln() + 1.0;
            sim.scale = scale;
            
            let value = sim.regularized_value(tree);
            trajectory.push(value);
        }

        trajectory
    }
}

fn tree_id(tree: &Tree) -> u64 {
    crate::utils::tree_hash(tree)
}

/// Generate dataset for learning renormalization patterns
pub fn generate_renorm_dataset(max_size: usize) -> Vec<(Tree, f32, f32)> {
    let mut hopf_ml = HopfML::new();
    let mut dataset = Vec::new();
    let sim = RenormalizationSimulator::new(0.1, 1.0);

    for size in 1..=max_size {
        let trees = hopf_ml.get_trees(size).clone();
        
        for tree in trees {
            let bare = sim.bare_value(&tree);
            let renorm = sim.regularized_value(&tree);
            dataset.push((tree, bare, renorm));
        }
    }

    dataset
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::TreeBuilder;

    #[test]
    fn test_divergence_degree() {
        let sim = RenormalizationSimulator::new(0.1, 1.0);
        
        let tree = Tree::new();
        assert!(sim.divergence_degree(&tree) <= 0);
        
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1);
        let tree2 = builder.build().unwrap();
        let div2 = sim.divergence_degree(&tree2);
        assert_eq!(div2, 0); // 2 nodes, height 1: 2 - 2*1 = 0
    }

    #[test]
    fn test_renormalization() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1)
            .add_child(0, 2)
            .add_child(1, 3);
        let tree = builder.build().unwrap();

        let sim = RenormalizationSimulator::new(0.1, 1.0);
        let bare = sim.bare_value(&tree);
        let renorm = sim.regularized_value(&tree);
        
        // Renormalized value should be different from bare
        assert!((bare - renorm).abs() > 1e-6);
    }

    #[test]
    fn test_rg_flow() {
        let tree = Tree::new();
        let flow = RenormalizationAnalyzer::rg_flow(&tree, 10);
        
        assert_eq!(flow.len(), 10);
        
        // Flow should be monotonic for simple tree
        for i in 1..flow.len() {
            assert!(flow[i] <= flow[i-1] * 1.1); // Allow small fluctuations
        }
    }
} 