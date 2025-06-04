//! Hopf-algebraic loss functions and regularizers

use std::sync::Arc;
use crate::algebra::{Tree, Forest, HopfAlgebra};
use crate::Result;
use ndarray::{Array1, Array2};

/// Regularizer that enforces Hopf algebra properties in embeddings
pub struct HopfRegularizer {
    /// Dimension of embeddings
    embedding_dim: usize,
    /// Weight for product linearity loss
    lambda_product: f32,
    /// Weight for antipode equivariance loss
    lambda_antipode: f32,
    /// Weight for coassociativity loss
    lambda_coassoc: f32,
}

impl HopfRegularizer {
    /// Create a new Hopf regularizer
    pub fn new(embedding_dim: usize) -> Self {
        HopfRegularizer {
            embedding_dim,
            lambda_product: 1.0,
            lambda_antipode: 1.0,
            lambda_coassoc: 0.5,
        }
    }
    
    /// Set the regularization weights
    pub fn with_weights(mut self, product: f32, antipode: f32, coassoc: f32) -> Self {
        self.lambda_product = product;
        self.lambda_antipode = antipode;
        self.lambda_coassoc = coassoc;
        self
    }

    /// Compute the regularization loss for a batch of tree embeddings
    /// 
    /// Returns (total_loss, components) where components has:
    /// - product_loss: linearity of multiplication
    /// - antipode_loss: involution property
    /// - coassociativity_loss: consistency of coproduct
    pub fn regularization_loss(
        &self,
        trees: &[Tree],
        embeddings: &Array2<f32>,
        hopf: Arc<HopfAlgebra>,
    ) -> (f32, Vec<f32>) {
        let mut product_loss = 0.0;
        let mut antipode_loss = 0.0;
        let mut coassoc_loss = 0.0;
        let _n_pairs = 0;
        
        // 1. Product linearity: φ(t₁ ⊙ t₂) ≈ φ(t₁) + φ(t₂)
        for i in 0..trees.len() {
            for j in i+1..trees.len() {
                // Create forest and compute embedding
                let forest = Forest::from_trees(vec![trees[i].clone(), trees[j].clone()]);
                let forest_emb = self.embed_forest(&forest, hopf.clone());
                
                // Compare with sum of individual embeddings
                let sum_emb = &embeddings.row(i) + &embeddings.row(j);
                let diff = &forest_emb - &sum_emb;
                product_loss += diff.dot(&diff) / self.embedding_dim as f32;
            }
        }
        
        // 2. Antipode involution: S(S(t)) = t
        let mut involution_loss = 0.0;
        for (i, tree) in trees.iter().enumerate() {
            // Apply antipode twice
            let s_tree = hopf.antipode(tree);
            let s_s_tree = hopf.antipode(&s_tree);
            
            // Check if S(S(t)) = t
            if tree != &s_s_tree {
                // Trees differ structurally - add penalty
                involution_loss += 10.0;
            } else {
                // Trees are the same, check embedding consistency
                let s_emb = self.embed_tree(&s_tree, hopf.clone());
                let s_s_emb = self.embed_tree(&s_s_tree, hopf.clone());
                
                // S(S(φ(t))) should equal φ(t)
                let orig_emb = embeddings.row(i);
                let diff = &s_s_emb - &orig_emb;
                involution_loss += diff.dot(&diff) / self.embedding_dim as f32;
            }
        }
        antipode_loss = involution_loss / trees.len() as f32;
        
        // 3. Coassociativity: (Δ ⊗ id) ∘ Δ = (id ⊗ Δ) ∘ Δ
        for tree in trees.iter() {
            let coassoc_err = self.check_coassociativity(tree, hopf.clone());
            coassoc_loss += coassoc_err;
        }
        coassoc_loss /= trees.len() as f32;
        
        let total_loss = self.lambda_product * product_loss + 
                        self.lambda_antipode * antipode_loss +
                        self.lambda_coassoc * coassoc_loss;
                        
        (total_loss, vec![product_loss, antipode_loss, coassoc_loss])
    }
    
    /// Helper to embed a forest by summing tree embeddings
    fn embed_forest(&self, forest: &Forest, hopf: Arc<HopfAlgebra>) -> Array1<f32> {
        let mut result = Array1::zeros(self.embedding_dim);
        for tree in forest.trees() {
            let tree_emb = self.embed_tree(tree, hopf.clone());
            result = result + tree_emb;
        }
        result
    }
    
    /// Helper to embed a single tree (placeholder - would use neural network)
    fn embed_tree(&self, tree: &Tree, _hopf: Arc<HopfAlgebra>) -> Array1<f32> {
        // In practice, this would use a neural network
        // For now, use a simple hash-based embedding
        let seed = tree.size() * 1000 + tree.max_depth() * 100 + tree.leaf_count();
        let mut result = Array1::zeros(self.embedding_dim);
        
        use rand::SeedableRng;
        let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
        for i in 0..self.embedding_dim {
            result[i] = rand::Rng::gen_range(&mut seeded_rng, -1.0..1.0);
        }
        
        // Normalize
        let norm = result.dot(&result).sqrt();
        if norm > 0.0 {
            result /= norm;
        }
        
        result
    }
    
    /// Check coassociativity property
    fn check_coassociativity(&self, tree: &Tree, hopf: Arc<HopfAlgebra>) -> f32 {
        // Compute Δ(t)
        let delta_t = hopf.coproduct(tree);
        
        let mut error = 0.0;
        
        // For each term in Δ(t) = Σ t' ⊗ t''
        for (forest1, forest2) in delta_t.terms() {
            // Compute (Δ ⊗ id)(t' ⊗ t'')
            let left_side = self.delta_tensor_id(forest1, forest2, hopf.clone());
            
            // Compute (id ⊗ Δ)(t' ⊗ t'')  
            let right_side = self.id_tensor_delta(forest1, forest2, hopf.clone());
            
            // These should be equal
            error += self.compare_tensor_products(&left_side, &right_side);
        }
        
        error
    }
    
    /// Apply (Δ ⊗ id) to a tensor product
    fn delta_tensor_id(
        &self, 
        forest1: &Forest, 
        forest2: &Forest,
        hopf: Arc<HopfAlgebra>
    ) -> Vec<(Forest, Forest, Forest)> {
        let mut result = Vec::new();
        
        // Apply Δ to first component
        for tree in forest1.trees() {
            let delta = hopf.coproduct(tree);
            for (f1, f2) in delta.terms() {
                // Result is f1 ⊗ f2 ⊗ forest2
                result.push((f1.clone(), f2.clone(), forest2.clone()));
            }
        }
        
        result
    }
    
    /// Apply (id ⊗ Δ) to a tensor product
    fn id_tensor_delta(
        &self,
        forest1: &Forest,
        forest2: &Forest, 
        hopf: Arc<HopfAlgebra>
    ) -> Vec<(Forest, Forest, Forest)> {
        let mut result = Vec::new();
        
        // Apply Δ to second component
        for tree in forest2.trees() {
            let delta = hopf.coproduct(tree);
            for (f1, f2) in delta.terms() {
                // Result is forest1 ⊗ f1 ⊗ f2
                result.push((forest1.clone(), f1.clone(), f2.clone()));
            }
        }
        
        result
    }
    
    /// Compare two triple tensor products
    fn compare_tensor_products(
        &self,
        left: &[(Forest, Forest, Forest)],
        right: &[(Forest, Forest, Forest)]
    ) -> f32 {
        // In a proper implementation, we'd check if the multisets are equal
        // For now, return a simple size mismatch penalty
        (left.len() as f32 - right.len() as f32).abs()
    }
} 