//! Antipode operation on rooted trees

use super::{Tree, Forest, CoProduct};
use std::collections::HashMap;
use std::sync::Mutex;
use once_cell::sync::Lazy;

/// Global memoization cache for antipode computations
static ANTIPODE_CACHE: Lazy<Mutex<HashMap<Tree, Forest>>> = 
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Trait for computing the antipode
pub trait Antipode {
    /// Compute the antipode S(t)
    /// For a tree t: S(t) = -t - Σ_{C proper} S(P_C(t)) · R_C(t)
    fn antipode(&self) -> Forest;
    
    /// Compute antipode with explicit memoization
    fn antipode_with_cache(&self, cache: &mut HashMap<Tree, Forest>) -> Forest;
}

impl Antipode for Tree {
    fn antipode(&self) -> Forest {
        let mut cache = ANTIPODE_CACHE.lock().unwrap();
        self.antipode_with_cache(&mut cache)
    }
    
    fn antipode_with_cache(&self, cache: &mut HashMap<Tree, Forest>) -> Forest {
        // Check cache first
        if let Some(cached) = cache.get(self) {
            return cached.clone();
        }
        
        // Base case: single node tree
        if self.size() == 1 {
            let result = Forest::single(self.clone());
            cache.insert(self.clone(), result.clone());
            return result;
        }
        
        // Start with -t
        let mut terms: HashMap<Forest, i32> = HashMap::new();
        terms.insert(Forest::single(self.clone()), -1);
        
        // Sum over all proper admissible cuts
        for cut in self.admissible_cuts() {
            if !cut.pruned_forest.is_empty() {
                // Compute S(P_C(t))
                let sp_c = antipode_forest(&cut.pruned_forest, cache);
                
                // Multiply by R_C(t)
                for (forest, coeff) in sp_c {
                    let combined = forest.multiply(&Forest::single(cut.trunk.clone()));
                    *terms.entry(combined).or_insert(0) += coeff;
                }
            }
        }
        
        // Collect non-zero terms into a forest
        let mut result_trees = Vec::new();
        for (forest, coeff) in terms {
            let abs_coeff = coeff.abs() as usize;
            for _ in 0..abs_coeff {
                result_trees.extend(forest.trees().iter().cloned());
            }
        }
        
        let result = Forest::from(result_trees);
        cache.insert(self.clone(), result.clone());
        result
    }
}

impl Antipode for Forest {
    fn antipode(&self) -> Forest {
        let mut cache = ANTIPODE_CACHE.lock().unwrap();
        self.antipode_with_cache(&mut cache)
    }
    
    fn antipode_with_cache(&self, cache: &mut HashMap<Tree, Forest>) -> Forest {
        antipode_forest(self, cache)
            .into_iter()
            .flat_map(|(f, coeff)| {
                let abs_coeff = coeff.abs() as usize;
                let mut trees = Vec::new();
                for _ in 0..abs_coeff {
                    trees.extend(f.trees().iter().cloned());
                }
                trees
            })
            .collect()
    }
}

/// Compute antipode of a forest (product of antipodes)
fn antipode_forest(forest: &Forest, cache: &mut HashMap<Tree, Forest>) -> HashMap<Forest, i32> {
    if forest.is_empty() {
        // S(empty) = empty
        let mut result = HashMap::new();
        result.insert(Forest::empty(), 1);
        return result;
    }
    
    // S(t1 × t2 × ...) = S(t1) × S(t2) × ...
    let mut result = HashMap::new();
    result.insert(Forest::empty(), 1);
    
    for tree in forest.iter() {
        let s_tree = tree.antipode_with_cache(cache);
        let mut next_result = HashMap::new();
        
        // Multiply current result by S(tree)
        for (f1, c1) in result {
            // S(tree) is a forest, treat it as coefficient 1 for each tree
            let combined = f1.multiply(&s_tree);
            *next_result.entry(combined).or_insert(0) += c1;
        }
        
        result = next_result;
    }
    
    result
}

/// Generate δ_k = sum of all rooted trees with k vertices
pub fn delta_k(k: usize) -> Vec<Tree> {
    use std::collections::BTreeSet;
    
    if k == 0 {
        return vec![];
    }
    if k == 1 {
        return vec![Tree::new()];
    }
    
    // Generate by grafting from smaller trees
    let prev = delta_k(k - 1);
    let mut result = BTreeSet::new();
    
    for tree in prev {
        for grafted in tree.graft_all_leaves() {
            result.insert(grafted);
        }
    }
    
    result.into_iter().collect()
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