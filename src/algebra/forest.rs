//! Forest: a commutative product of rooted trees

use super::Tree;
use std::fmt;
use serde::{Serialize, Deserialize};
use std::hash::Hash;

/// A forest is a commutative product of trees, stored as a sorted vector
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Forest {
    trees: Vec<Tree>,
}

impl Forest {
    /// Create an empty forest (identity element)
    pub fn empty() -> Self {
        Forest { trees: Vec::new() }
    }

    /// Create a forest from a vector of trees
    pub fn from(mut trees: Vec<Tree>) -> Self {
        trees.sort();
        Forest { trees }
    }

    /// Create a forest with a single tree
    pub fn single(tree: Tree) -> Self {
        Forest { trees: vec![tree] }
    }

    /// Check if forest is empty
    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    /// Get the number of trees
    pub fn len(&self) -> usize {
        self.trees.len()
    }

    /// Get the trees
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Get the total number of nodes across all trees
    pub fn total_nodes(&self) -> usize {
        self.trees.iter().map(|t| t.size()).sum()
    }

    /// Multiply two forests (concatenate and sort)
    pub fn multiply(&self, other: &Forest) -> Forest {
        let mut trees = self.trees.clone();
        trees.extend(other.trees.clone());
        trees.sort();
        Forest { trees }
    }

    /// Apply a function to each tree and collect results
    pub fn map<F, R>(&self, f: F) -> Vec<R>
    where
        F: Fn(&Tree) -> R,
    {
        self.trees.iter().map(f).collect()
    }

    /// Create iterator over trees
    pub fn iter(&self) -> std::slice::Iter<Tree> {
        self.trees.iter()
    }
}

impl fmt::Debug for Forest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.trees.is_empty() {
            write!(f, "∅")
        } else {
            write!(f, "Forest[")?;
            for (i, tree) in self.trees.iter().enumerate() {
                if i > 0 {
                    write!(f, " × ")?;
                }
                write!(f, "T{}", tree.size())?;
            }
            write!(f, "]")
        }
    }
}

impl Default for Forest {
    fn default() -> Self {
        Self::empty()
    }
}

impl From<Tree> for Forest {
    fn from(tree: Tree) -> Self {
        Forest::single(tree)
    }
}

impl FromIterator<Tree> for Forest {
    fn from_iter<I: IntoIterator<Item = Tree>>(iter: I) -> Self {
        let mut trees: Vec<Tree> = iter.into_iter().collect();
        trees.sort();
        Forest { trees }
    }
}

impl IntoIterator for Forest {
    type Item = Tree;
    type IntoIter = std::vec::IntoIter<Tree>;

    fn into_iter(self) -> Self::IntoIter {
        self.trees.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_forest() {
        let f = Forest::empty();
        assert!(f.is_empty());
        assert_eq!(f.len(), 0);
    }

    #[test]
    fn test_forest_multiplication() {
        let t1 = Tree::new();
        let t2 = Tree::new();
        
        let f1 = Forest::single(t1);
        let f2 = Forest::single(t2);
        
        let f3 = f1.multiply(&f2);
        assert_eq!(f3.len(), 2);
    }
} 