//! Utility functions for Hopf-ML

mod latex_parser;

use crate::algebra::Tree;
use std::fs::File;
use std::io::{Write, Read};
use serde::{Serialize, Deserialize};

pub use latex_parser::{LaTeXParser, MathExpr, latex_to_tree};

/// Save object to JSON file
pub fn save_json<T: Serialize>(obj: &T, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(obj)?;
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

/// Load object from JSON file
pub fn load_json<T: for<'de> Deserialize<'de>>(path: &str) -> Result<T, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let obj = serde_json::from_str(&contents)?;
    Ok(obj)
}

/// Generate a random tree of given size
pub fn random_tree(size: usize) -> Tree {
    use rand::Rng;
    
    if size == 0 {
        panic!("Cannot create tree with 0 nodes");
    }
    
    if size == 1 {
        return Tree::new();
    }
    
    let mut rng = rand::thread_rng();
    let mut children = vec![Vec::new(); size];
    
    // Build tree by adding nodes one by one
    for node in 1..size {
        // Choose random parent from existing nodes
        let parent = rng.gen_range(0..node);
        children[parent].push(node);
    }
    
    Tree::from_adjacency(children).unwrap()
}

/// Pretty print a tree
pub fn print_tree(tree: &Tree) {
    println!("{:?}", tree);
}

/// Compute tree hash for deduplication
pub fn tree_hash(tree: &Tree) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let canonical = tree_canonical_string(tree);
    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    hasher.finish()
}

/// Get canonical string representation of tree
fn tree_canonical_string(tree: &Tree) -> String {
    fn recursive(tree: &Tree, node: usize) -> String {
        let children = tree.children(node);
        if children.is_empty() {
            "()".to_string()
        } else {
            let mut child_strs: Vec<String> = children
                .iter()
                .map(|&child| recursive(tree, child))
                .collect();
            child_strs.sort();
            format!("({})", child_strs.join(""))
        }
    }
    
    recursive(tree, 0)
}

/// Timing utilities
pub mod timing {
    use std::time::Instant;
    
    /// Simple timer
    pub struct Timer {
        start: Instant,
        name: String,
    }
    
    impl Timer {
        /// Start new timer
        pub fn new(name: &str) -> Self {
            Timer {
                start: Instant::now(),
                name: name.to_string(),
            }
        }
        
        /// Get elapsed time
        pub fn elapsed(&self) -> f32 {
            self.start.elapsed().as_secs_f32()
        }
        
        /// Print elapsed time
        pub fn print(&self) {
            println!("{}: {:.3}s", self.name, self.elapsed());
        }
    }
    
    impl Drop for Timer {
        fn drop(&mut self) {
            self.print();
        }
    }
}

/// Progress tracking
pub mod progress {
    use std::io::{self, Write};
    
    /// Simple progress bar
    pub struct ProgressBar {
        total: usize,
        current: usize,
        width: usize,
    }
    
    impl ProgressBar {
        /// Create new progress bar
        pub fn new(total: usize) -> Self {
            ProgressBar {
                total,
                current: 0,
                width: 50,
            }
        }
        
        /// Update progress
        pub fn update(&mut self, current: usize) {
            self.current = current;
            self.display();
        }
        
        /// Increment progress
        pub fn inc(&mut self) {
            self.current += 1;
            self.display();
        }
        
        /// Display progress bar
        fn display(&self) {
            let progress = self.current as f32 / self.total as f32;
            let filled = (progress * self.width as f32) as usize;
            let empty = self.width - filled;
            
            print!("\r[");
            print!("{}", "=".repeat(filled));
            print!("{}", " ".repeat(empty));
            print!("] {}/{} ({:.1}%)", self.current, self.total, progress * 100.0);
            
            if self.current >= self.total {
                println!();
            }
            
            io::stdout().flush().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_random_tree() {
        for size in 1..=5 {
            let tree = random_tree(size);
            assert_eq!(tree.size(), size);
        }
    }
    
    #[test]
    fn test_tree_hash() {
        let tree1 = Tree::new();
        let tree2 = Tree::new();
        assert_eq!(tree_hash(&tree1), tree_hash(&tree2));
        
        let tree3 = random_tree(5);
        let tree4 = random_tree(5);
        // Very unlikely to be the same
        assert_ne!(tree_hash(&tree3), tree_hash(&tree4));
    }
    
    #[test]
    fn test_json_serialization() {
        let tree = random_tree(3);
        let path = "/tmp/test_tree.json";
        
        save_json(&tree, path).unwrap();
        let loaded: Tree = load_json(path).unwrap();
        
        assert_eq!(tree, loaded);
        
        // Clean up
        std::fs::remove_file(path).ok();
    }
} 