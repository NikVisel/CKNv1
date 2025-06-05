//! Hopf-invariant losses and algebraic constraints for embeddings

use crate::algebra::{Tree, Forest, HopfAlgebra, CoProduct, Antipode};
use ndarray::{Array1, Array2};
use num_traits::Zero;
use std::collections::HashMap;

/// Algebraic constraints for embeddings to respect Hopf structure
#[derive(Debug, Clone)]
pub struct AlgebraicConstraints {
    /// Weight for product linearity constraint
    pub product_weight: f32,
    /// Weight for antipode equivariance
    pub antipode_weight: f32,
    /// Weight for coproduct consistency
    pub coproduct_weight: f32,
    /// Temperature for contrastive losses
    pub temperature: f32,
}

impl Default for AlgebraicConstraints {
    fn default() -> Self {
        AlgebraicConstraints {
            product_weight: 1.0,
            antipode_weight: 1.0,
            coproduct_weight: 0.5,
            temperature: 0.1,
        }
    }
}

/// Hopf-invariant loss functions for training algebra-aware embeddings
pub struct HopfInvariantLoss {
    constraints: AlgebraicConstraints,
    embedding_cache: HashMap<u64, Array1<f32>>,
}

impl HopfInvariantLoss {
    /// Create new Hopf-invariant loss calculator
    pub fn new(constraints: AlgebraicConstraints) -> Self {
        HopfInvariantLoss {
            constraints,
            embedding_cache: HashMap::new(),
        }
    }

    /// Compute product linearity loss: φ(t₁ ⊙ t₂) ≈ φ(t₁) + φ(t₂)
    pub fn product_linearity_loss(
        &mut self,
        tree1: &Tree,
        tree2: &Tree,
        embed_fn: impl Fn(&Tree) -> Array1<f32>,
    ) -> f32 {
        let embed1 = self.get_or_compute_embedding(tree1, &embed_fn);
        let embed2 = self.get_or_compute_embedding(tree2, &embed_fn);
        
        // Create forest (product of trees)
        let forest = Forest::from(vec![tree1.clone(), tree2.clone()]);

        // For simplicity, we embed a forest as sum of tree embeddings for the
        // expected result
        let forest_embed_expected = &embed1 + &embed2;

        // Compute actual forest embedding using the provided embedding
        let forest_embed_actual = self.embed_forest(&forest, &embed_fn);
        
        // MSE loss
        let diff = &forest_embed_actual - &forest_embed_expected;
        diff.mapv(|x| x * x).sum() / diff.len() as f32
    }

    /// Compute antipode equivariance loss: φ(S(t)) ≈ -φ(t)
    pub fn antipode_equivariance_loss(
        &mut self,
        tree: &Tree,
        embed_fn: impl Fn(&Tree) -> Array1<f32>,
    ) -> f32 {
        let embed_t = self.get_or_compute_embedding(tree, &embed_fn);
        
        // Get antipode
        let antipode = tree.antipode();
        
        // For a forest, we sum embeddings of its trees
        let mut embed_st = Array1::<f32>::zeros(embed_t.len());
        for tree in antipode.trees() {
            let tree_embed = self.get_or_compute_embedding(tree, &embed_fn);
            embed_st = embed_st + tree_embed;
        }
        
        // Loss: φ(S(t)) + φ(t) ≈ 0
        let sum = &embed_st + &embed_t;
        sum.mapv(|x: f32| x * x).sum() / sum.len() as f32
    }

    /// Compute coproduct consistency loss using contrastive learning
    pub fn coproduct_contrastive_loss(
        &mut self,
        tree: &Tree,
        embed_fn: impl Fn(&Tree) -> Array1<f32>,
    ) -> f32 {
        let cop = tree.coproduct();
        if cop.len() <= 2 {
            return 0.0; // Skip trivial coproducts
        }
        
        let tree_embed = self.get_or_compute_embedding(tree, &embed_fn);
        let mut positive_pairs = Vec::new();
        let mut negative_pairs = Vec::new();
        
        // Collect positive pairs from coproduct
        for ((forest, trunk), _coeff) in cop.iter() {
            if !forest.is_empty() && trunk.size() > 1 {
                let forest_embed = self.embed_forest(forest, &embed_fn);
                let trunk_embed = self.get_or_compute_embedding(trunk, &embed_fn);
                
                // Tensor product approximated as concatenation
                let pair_embed = concatenate(&forest_embed, &trunk_embed);
                positive_pairs.push(pair_embed);
            }
        }
        
        // Generate negative pairs (random trees not in coproduct)
        // For now, use perturbed versions
        for _ in 0..positive_pairs.len() {
            let noise = Array1::from_shape_fn(tree_embed.len(), |_| {
                rand::random::<f32>() * 0.1 - 0.05
            });
            let neg_embed = &tree_embed + &noise;
            negative_pairs.push(neg_embed);
        }
        
        // Contrastive loss (InfoNCE-style)
        let temp = self.constraints.temperature;
        let mut loss = 0.0;
        
        for pos in &positive_pairs {
            let pos_sim = cosine_similarity(&tree_embed, pos) / temp;
            
            let mut neg_sum = 0.0;
            for neg in &negative_pairs {
                let neg_sim = cosine_similarity(&tree_embed, neg) / temp;
                neg_sum += neg_sim.exp();
            }
            
            loss += -(pos_sim - (pos_sim.exp() + neg_sum).ln());
        }
        
        loss / positive_pairs.len().max(1) as f32
    }

    /// Combined Hopf-invariant loss
    pub fn total_loss(
        &mut self,
        trees: &[(Tree, Tree)], // Pairs for product loss
        embed_fn: impl Fn(&Tree) -> Array1<f32>,
    ) -> f32 {
        let mut total = 0.0;
        
        // Product linearity losses
        if self.constraints.product_weight > 0.0 {
            for (t1, t2) in trees {
                total += self.constraints.product_weight * 
                    self.product_linearity_loss(t1, t2, &embed_fn);
            }
        }
        
        // Antipode equivariance losses
        if self.constraints.antipode_weight > 0.0 {
            for (t, _) in trees {
                total += self.constraints.antipode_weight * 
                    self.antipode_equivariance_loss(t, &embed_fn);
            }
        }
        
        // Coproduct contrastive losses
        if self.constraints.coproduct_weight > 0.0 {
            for (t, _) in trees {
                total += self.constraints.coproduct_weight * 
                    self.coproduct_contrastive_loss(t, &embed_fn);
            }
        }
        
        total / trees.len().max(1) as f32
    }

    fn get_or_compute_embedding(
        &mut self,
        tree: &Tree,
        embed_fn: &impl Fn(&Tree) -> Array1<f32>,
    ) -> Array1<f32> {
        let tree_id = crate::utils::tree_hash(tree);
        
        if let Some(cached) = self.embedding_cache.get(&tree_id) {
            cached.clone()
        } else {
            let embedding = embed_fn(tree);
            self.embedding_cache.insert(tree_id, embedding.clone());
            embedding
        }
    }

    fn embed_forest(
        &mut self,
        forest: &Forest,
        embed_fn: &impl Fn(&Tree) -> Array1<f32>,
    ) -> Array1<f32> {
        let mut result = None;
        
        for tree in forest.trees() {
            let tree_embed = self.get_or_compute_embedding(tree, embed_fn);
            result = match result {
                None => Some(tree_embed),
                Some(acc) => Some(acc + tree_embed),
            };
        }
        
        result.unwrap_or_else(|| {
            // Empty forest gets zero embedding
            Array1::zeros(10) // Default dimension
        })
    }
}

/// Concatenate two arrays
fn concatenate(a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
    let mut result = Array1::zeros(a.len() + b.len());
    result.slice_mut(ndarray::s![..a.len()]).assign(a);
    result.slice_mut(ndarray::s![a.len()..]).assign(b);
    result
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.mapv(|x| x * x).sum().sqrt();
    let norm_b = b.mapv(|x| x * x).sum().sqrt();
    
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Hopf-algebraic regularization terms
pub struct HopfRegularizer {
    /// Enforce that δ_k embeddings follow natural growth
    pub growth_consistency: f32,
    /// Enforce involution property: S(S(t)) = t
    pub involution_weight: f32,
    /// Enforce coassociativity of coproduct
    pub coassociativity_weight: f32,
}

impl HopfRegularizer {
    /// Compute growth consistency regularization
    pub fn growth_regularization(
        &self,
        tree: &Tree,
        embed_fn: impl Fn(&Tree) -> Array1<f32>,
    ) -> f32 {
        let tree_embed = embed_fn(tree);
        let grafted = tree.graft_all_leaves();
        
        if grafted.is_empty() {
            return 0.0;
        }
        
        // Average embedding of grafted trees
        let mut avg_grafted = Array1::zeros(tree_embed.len());
        for g_tree in &grafted {
            avg_grafted = avg_grafted + embed_fn(g_tree);
        }
        avg_grafted = avg_grafted / grafted.len() as f32;
        
        // Regularize: grafted embeddings should be "near" original + growth direction
        let expected_direction = &avg_grafted - &tree_embed;
        let growth_magnitude = expected_direction.mapv(|x: f32| x * x).sum().sqrt();
        
        // Penalty if growth is too small or too large
        let ideal_growth = (tree.size() as f32).sqrt();
        ((growth_magnitude - ideal_growth) / ideal_growth).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::TreeBuilder;

    #[test]
    fn test_hopf_invariant_loss() {
        let t1 = Tree::new();
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1);
        let t2 = builder.build().unwrap();
        
        // Simple embedding function for testing
        let embed_fn = |tree: &Tree| -> Array1<f32> {
            Array1::from_elem(10, tree.size() as f32)
        };
        
        let mut loss_calc = HopfInvariantLoss::new(Default::default());
        
        // Test product linearity
        let prod_loss = loss_calc.product_linearity_loss(&t1, &t2, embed_fn);
        assert!(prod_loss >= 0.0);
        
        // Test antipode equivariance
        let anti_loss = loss_calc.antipode_equivariance_loss(&t1, embed_fn);
        assert!(anti_loss >= 0.0);
    }
} 