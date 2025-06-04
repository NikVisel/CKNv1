//! Graph generation using Hopf algebra structures

use crate::algebra::Tree;

/// Generator for trees using Hopf-algebraic principles
pub struct HopfTreeGenerator;

impl HopfTreeGenerator {
    /// Generate trees by iterative grafting
    pub fn generate_by_grafting(start: &Tree, iterations: usize) -> Vec<Tree> {
        let mut current = vec![start.clone()];
        
        for _ in 0..iterations {
            let mut next = Vec::new();
            for tree in &current {
                next.extend(tree.graft_all_leaves());
            }
            current = next;
        }
        
        current
    }
} 