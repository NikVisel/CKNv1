//! Coproduct operation on rooted trees

use super::{Tree, Forest};
use std::collections::{HashSet, HashMap};
use rayon::prelude::*;

/// Represents an admissible cut of a tree
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissibleCut {
    /// Edges to cut (parent, child) pairs
    pub cut_edges: HashSet<(usize, usize)>,
    /// The pruned subtrees (forest)
    pub pruned_forest: Forest,
    /// The remaining trunk
    pub trunk: Tree,
}

/// Trait for computing coproduct
pub trait CoProduct {
    /// Compute all admissible cuts
    fn admissible_cuts(&self) -> Vec<AdmissibleCut>;
    
    /// Compute the full coproduct as (Forest, Tree) pairs with coefficients
    fn coproduct(&self) -> HashMap<(Forest, Tree), i32>;
}

impl CoProduct for Tree {
    fn admissible_cuts(&self) -> Vec<AdmissibleCut> {
        // Collect all edges
        let mut edges = Vec::new();
        for parent in 0..self.size() {
            for &child in self.children(parent) {
                edges.push((parent, child));
            }
        }
        
        let n_edges = edges.len();
        
        // For large trees, parallelize the search
        let use_parallel = n_edges > 8;
        
        let masks: Vec<usize> = (0..(1 << n_edges)).collect();
        let cuts = if use_parallel {
            masks.par_iter()
                .filter_map(|&mask| self.process_cut_mask(mask, &edges))
                .collect()
        } else {
            masks.iter()
                .filter_map(|&mask| self.process_cut_mask(mask, &edges))
                .collect()
        };
        
        cuts
    }
    
    fn coproduct(&self) -> HashMap<(Forest, Tree), i32> {
        let mut result = HashMap::new();
        
        // Empty cut: ∅ ⊗ t
        result.insert((Forest::empty(), self.clone()), 1);
        
        // Full cut: t ⊗ 1
        result.insert((Forest::from(vec![self.clone()]), Tree::new()), 1);
        
        // All proper admissible cuts
        for cut in self.admissible_cuts() {
            if !cut.pruned_forest.is_empty() && cut.trunk.size() > 1 {
                result.insert((cut.pruned_forest, cut.trunk), 1);
            }
        }
        
        result
    }
}

impl Tree {
    fn process_cut_mask(&self, mask: usize, edges: &[(usize, usize)]) -> Option<AdmissibleCut> {
        // Collect cut edges
        let mut cut_edges = HashSet::new();
        for (i, &edge) in edges.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                cut_edges.insert(edge);
            }
        }
        
        if cut_edges.is_empty() {
            return None;
        }
        
        // Check admissibility: no two cuts on same root-to-leaf path
        if !self.is_admissible_cut(&cut_edges) {
            return None;
        }
        
        // Build forest and trunk
        let (forest, trunk) = self.build_forest_and_trunk(&cut_edges);
        
        Some(AdmissibleCut {
            cut_edges,
            pruned_forest: forest,
            trunk,
        })
    }
    
    fn is_admissible_cut(&self, cut_edges: &HashSet<(usize, usize)>) -> bool {
        let paths = self.root_to_leaf_paths();
        
        for path in paths {
            let mut cuts_on_path = 0;
            
            for i in 0..path.len() - 1 {
                let edge = (path[i], path[i + 1]);
                if cut_edges.contains(&edge) {
                    cuts_on_path += 1;
                    if cuts_on_path > 1 {
                        return false;
                    }
                }
            }
        }
        
        true
    }
    
    fn build_forest_and_trunk(&self, cut_edges: &HashSet<(usize, usize)>) -> (Forest, Tree) {
        // Find nodes in trunk (connected to root after cuts)
        let mut in_trunk = vec![false; self.size()];
        let mut stack = vec![0];
        in_trunk[0] = true;
        
        while let Some(node) = stack.pop() {
            for &child in self.children(node) {
                if !cut_edges.contains(&(node, child)) && !in_trunk[child] {
                    in_trunk[child] = true;
                    stack.push(child);
                }
            }
        }
        
        // Build trunk
        let trunk = self.build_subtree(&in_trunk);
        
        // Build pruned subtrees
        let mut forest_trees = Vec::new();
        let mut component_assigned = vec![false; self.size()];
        
        for v in 0..self.size() {
            if !in_trunk[v] && !component_assigned[v] {
                // Find root of this component (node whose parent edge was cut)
                let component_root = v;
                
                // Mark all nodes in this component
                let mut component_nodes = vec![false; self.size()];
                let mut stack = vec![component_root];
                component_nodes[component_root] = true;
                component_assigned[component_root] = true;
                
                while let Some(node) = stack.pop() {
                    for &child in self.children(node) {
                        if !in_trunk[child] && !component_nodes[child] 
                            && !cut_edges.contains(&(node, child)) {
                            component_nodes[child] = true;
                            component_assigned[child] = true;
                            stack.push(child);
                        }
                    }
                }
                
                forest_trees.push(self.build_subtree(&component_nodes));
            }
        }
        
        (Forest::from(forest_trees), trunk)
    }
    
    fn build_subtree(&self, included_nodes: &[bool]) -> Tree {
        // Renumber nodes
        let mut old_to_new = HashMap::new();
        let mut new_index = 0;
        
        for (old, &included) in included_nodes.iter().enumerate() {
            if included {
                old_to_new.insert(old, new_index);
                new_index += 1;
            }
        }
        
        let n_nodes = new_index;
        let mut children = vec![Vec::new(); n_nodes];
        
        for (old_parent, &included) in included_nodes.iter().enumerate() {
            if included {
                let new_parent = old_to_new[&old_parent];
                for &old_child in self.children(old_parent) {
                    if included_nodes[old_child] {
                        let new_child = old_to_new[&old_child];
                        children[new_parent].push(new_child);
                    }
                }
            }
        }
        
        Tree::from_adjacency(children).expect("invalid adjacency")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::TreeBuilder;

    #[test]
    fn test_coproduct_single_node() {
        let t = Tree::new();
        let cop = t.coproduct();
        
        // Should have exactly 2 terms: ∅⊗t and t⊗1
        assert_eq!(cop.len(), 2);
    }

    #[test]
    fn test_coproduct_two_nodes() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1);
        let t = builder.build().unwrap();
        
        let cop = t.coproduct();
        
        // Should have 3 terms: ∅⊗t, t⊗1, and t₁⊗t₁
        assert_eq!(cop.len(), 3);
    }
} 