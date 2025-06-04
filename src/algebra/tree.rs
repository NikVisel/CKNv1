//! Rooted tree data structure and operations

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

/// A rooted tree represented as an adjacency list
/// 
/// - Node 0 is always the root
/// - Each node stores indices of its children
/// - Trees are ordered lexicographically for canonical representation
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tree {
    /// Number of nodes in the tree
    n_nodes: usize,
    /// Adjacency list: children[i] contains indices of node i's children
    children: Vec<Vec<usize>>,
}

impl Tree {
    /// Create a new tree with a single root node
    pub fn new() -> Self {
        Tree {
            n_nodes: 1,
            children: vec![Vec::new()],
        }
    }

    /// Create a tree with specified structure
    pub fn from_adjacency(children: Vec<Vec<usize>>) -> crate::Result<Self> {
        let n_nodes = children.len();
        
        // Validate structure
        for (parent, child_list) in children.iter().enumerate() {
            for &child in child_list {
                if child >= n_nodes {
                    return Err(crate::HopfMLError::InvalidTree(
                        format!("Child index {} out of bounds", child)
                    ));
                }
                if child == parent {
                    return Err(crate::HopfMLError::InvalidTree(
                        "Self-loops not allowed".to_string()
                    ));
                }
            }
        }
        
        // Check connectivity (BFS from root)
        let mut visited = vec![false; n_nodes];
        let mut queue = VecDeque::new();
        queue.push_back(0);
        visited[0] = true;
        
        while let Some(node) = queue.pop_front() {
            for &child in &children[node] {
                if !visited[child] {
                    visited[child] = true;
                    queue.push_back(child);
                }
            }
        }
        
        if visited.iter().any(|&v| !v) {
            return Err(crate::HopfMLError::InvalidTree(
                "Tree is not connected".to_string()
            ));
        }
        
        Ok(Tree { n_nodes, children })
    }

    /// Get the number of nodes
    pub fn size(&self) -> usize {
        self.n_nodes
    }

    /// Get children of a node
    pub fn children(&self, node: usize) -> &[usize] {
        &self.children[node]
    }

    /// Find parent of a node (None for root)
    pub fn parent(&self, node: usize) -> Option<usize> {
        if node == 0 {
            return None;
        }
        
        for (parent, children) in self.children.iter().enumerate() {
            if children.contains(&node) {
                return Some(parent);
            }
        }
        None
    }

    /// Compute depth of each node
    pub fn depths(&self) -> Vec<usize> {
        let mut depths = vec![0; self.n_nodes];
        let mut queue = VecDeque::new();
        queue.push_back((0, 0));
        
        while let Some((node, depth)) = queue.pop_front() {
            depths[node] = depth;
            for &child in &self.children[node] {
                queue.push_back((child, depth + 1));
            }
        }
        
        depths
    }

    /// Apply natural growth operator N: attach a new leaf at each node
    pub fn graft_all_leaves(&self) -> Vec<Tree> {
        let mut result = Vec::with_capacity(self.n_nodes);
        
        for attach_point in 0..self.n_nodes {
            let mut new_children = self.children.clone();
            new_children.push(Vec::new()); // New leaf has no children
            new_children[attach_point].push(self.n_nodes);
            
            let new_tree = Tree {
                n_nodes: self.n_nodes + 1,
                children: new_children,
            };
            result.push(new_tree);
        }
        
        result
    }

    /// Get all paths from root to leaves
    pub fn root_to_leaf_paths(&self) -> Vec<Vec<usize>> {
        let mut paths = Vec::new();
        let mut current_path = vec![0];
        self.dfs_paths(0, &mut current_path, &mut paths);
        paths
    }

    fn dfs_paths(&self, node: usize, current: &mut Vec<usize>, paths: &mut Vec<Vec<usize>>) {
        if self.children[node].is_empty() {
            // Leaf node
            paths.push(current.clone());
        } else {
            for &child in &self.children[node] {
                current.push(child);
                self.dfs_paths(child, current, paths);
                current.pop();
            }
        }
    }

    /// Check if this tree is isomorphic to another
    pub fn is_isomorphic(&self, other: &Tree) -> bool {
        if self.n_nodes != other.n_nodes {
            return false;
        }
        
        // For small trees, we can use canonical form comparison
        self.canonical_form() == other.canonical_form()
    }

    /// Compute a canonical form for isomorphism checking
    fn canonical_form(&self) -> String {
        self.canonical_form_recursive(0)
    }

    fn canonical_form_recursive(&self, node: usize) -> String {
        let children = &self.children[node];
        if children.is_empty() {
            return "()".to_string();
        }
        
        let mut child_forms: Vec<String> = children
            .iter()
            .map(|&child| self.canonical_form_recursive(child))
            .collect();
        
        child_forms.sort();
        format!("({})", child_forms.join(""))
    }

    /// Maximum depth of the tree
    pub fn max_depth(&self) -> usize {
        self.compute_max_depth(0)
    }

    /// Count of leaf nodes
    pub fn leaf_count(&self) -> usize {
        (0..self.size())
            .filter(|&i| self.children(i).is_empty())
            .count()
    }

    fn compute_max_depth(&self, node: usize) -> usize {
        let children = self.children(node);
        if children.is_empty() {
            0
        } else {
            1 + children.iter()
                .map(|&child| self.compute_max_depth(child))
                .max()
                .unwrap_or(0)
        }
    }

    /// Compute the depth of a specific node
    pub fn node_depth(&self, node: usize) -> usize {
        if node == 0 {
            return 0;
        }
        
        if let Some(parent) = self.parent(node) {
            1 + self.node_depth(parent)
        } else {
            0
        }
    }
    
    /// Compute subtree size rooted at a given node
    pub fn subtree_size(&self, node: usize) -> usize {
        1 + self.children(node)
            .iter()
            .map(|&child| self.subtree_size(child))
            .sum::<usize>()
    }
    
    /// Get the path from a node to the root
    pub fn path_to_root(&self, mut node: usize) -> Vec<usize> {
        let mut path = vec![node];
        while node != 0 {
            if let Some(parent) = self.parent(node) {
                path.push(parent);
                node = parent;
            } else {
                break;
            }
        }
        path.reverse();
        path
    }
    
    /// Compute the "Hopf degree" - how many times this tree pattern appears
    /// when generating trees via natural growth
    pub fn hopf_degree(&self) -> usize {
        // For now, return a heuristic based on tree structure
        // In practice, this would require enumeration
        self.size() * self.max_depth() + self.leaf_count()
    }
    
    /// Get all nodes at a specific depth
    pub fn nodes_at_depth(&self, target_depth: usize) -> Vec<usize> {
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((0, 0));
        
        while let Some((node, depth)) = queue.pop_front() {
            if depth == target_depth {
                result.push(node);
            } else if depth < target_depth {
                for &child in self.children(node) {
                    queue.push_back((child, depth + 1));
                }
            }
        }
        
        result
    }
    
    /// Check if two nodes are siblings (share the same parent)
    pub fn are_siblings(&self, node1: usize, node2: usize) -> bool {
        if node1 == node2 || node1 == 0 || node2 == 0 {
            return false;
        }
        
        self.parent(node1) == self.parent(node2)
    }
    
    /// Get the degree (number of children) of a node
    pub fn node_degree(&self, node: usize) -> usize {
        self.children(node).len()
    }
}

/// Total ordering for trees (lexicographic on adjacency lists)
impl Ord for Tree {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.n_nodes.cmp(&other.n_nodes) {
            std::cmp::Ordering::Equal => {
                for i in 0..self.n_nodes {
                    match self.children[i].cmp(&other.children[i]) {
                        std::cmp::Ordering::Equal => continue,
                        other => return other,
                    }
                }
                std::cmp::Ordering::Equal
            }
            other => other,
        }
    }
}

impl PartialOrd for Tree {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Tree {
    fn fmt_recursive(&self, f: &mut fmt::Formatter<'_>, node: usize, indent: usize) -> fmt::Result {
        for _ in 0..indent {
            write!(f, "  ")?;
        }
        writeln!(f, "● {}", node)?;
        
        for &child in &self.children[node] {
            self.fmt_recursive(f, child, indent + 1)?;
        }
        Ok(())
    }
}

impl fmt::Debug for Tree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Tree(size={})", self.n_nodes)?;
        self.fmt_recursive(f, 0, 0)
    }
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing trees incrementally
pub struct TreeBuilder {
    children: Vec<Vec<usize>>,
}

impl TreeBuilder {
    /// Create a new builder starting with a root
    pub fn new() -> Self {
        TreeBuilder {
            children: vec![Vec::new()],
        }
    }

    /// Add a child to a parent node
    pub fn add_child(&mut self, parent: usize, child: usize) -> &mut Self {
        while self.children.len() <= child {
            self.children.push(Vec::new());
        }
        
        if parent < self.children.len() {
            self.children[parent].push(child);
        }
        
        self
    }

    /// Build the tree
    pub fn build(self) -> crate::Result<Tree> {
        Tree::from_adjacency(self.children)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_node() {
        let tree = Tree::new();
        assert_eq!(tree.size(), 1);
        let empty: &[usize] = &[];
        assert_eq!(tree.children(0), empty);
    }

    #[test]
    fn test_tree_builder() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1)
            .add_child(0, 2)
            .add_child(1, 3);
        let tree = builder.build().unwrap();
        
        assert_eq!(tree.size(), 4);
        assert_eq!(tree.children(0), &[1usize, 2]);
        assert_eq!(tree.children(1), &[3usize]);
    }

    #[test]
    fn test_grafting() {
        let tree = Tree::new();
        let grafted = tree.graft_all_leaves();
        assert_eq!(grafted.len(), 1);
        assert_eq!(grafted[0].size(), 2);
    }
} 