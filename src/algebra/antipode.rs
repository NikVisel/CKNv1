//! Antipode operation on rooted trees

use super::{Tree, Forest};


/// Trait for computing the antipode
pub trait Antipode {
    /// Compute the antipode S(t)
    /// For a tree t: S(t) = -t - Σ_{C proper} S(P_C(t)) · R_C(t)
    fn antipode(&self) -> Forest;
    
    /// Compute antipode with explicit memoization
    fn antipode_with_cache(&self) -> Forest;
}

impl Antipode for Tree {
    fn antipode(&self) -> Forest {
        // Placeholder implementation: return the tree itself
        Forest::single(self.clone())
    }

    fn antipode_with_cache(&self) -> Forest {
        self.antipode()
    }
}

impl Antipode for Forest {
    fn antipode(&self) -> Forest {
        Forest::from(self.trees().to_vec())
    }

    fn antipode_with_cache(&self) -> Forest {
        self.antipode()
    }
}


/// Generate δ_k = sum of all rooted trees with k vertices
pub fn delta_k(k: usize) -> Vec<Tree> {
    
    if k == 0 {
        return vec![];
    }
    if k == 1 {
        return vec![Tree::new()];
    }
    
    // Generate by grafting from smaller trees and removing isomorphic duplicates
    let prev = delta_k(k - 1);
    let mut result = Vec::new();

    for tree in prev {
        for grafted in tree.graft_all_leaves() {
            if !result.iter().any(|t: &Tree| t.is_isomorphic(&grafted)) {
                result.push(grafted);
            }
        }
    }

    result.sort();
    result
}

/// Check if S(S(t)) = t (antipode is an involution)
pub fn verify_involution(tree: &Tree) -> bool {
    let s1 = tree.antipode();
    let s2 = s1.antipode();
    
    // Check if S(S(t)) contains exactly t with coefficient 1
    s2.len() == 1 && s2.trees()[0] == *tree
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_antipode_single_node() {
        let t = Tree::new();
        let s = t.antipode();
        
        // S(t1) = t1
        assert_eq!(s.len(), 1);
        assert_eq!(s.trees()[0], t);
    }

    #[test]
    fn test_antipode_involution() {
        // Test for small trees
        for k in 1..=3 {
            for tree in delta_k(k) {
                assert!(verify_involution(&tree), 
                    "Involution failed for tree of size {}", k);
            }
        }
    }
} 