//! Main Hopf algebra structure and operations

use super::{Tree, Forest, CoProduct, Antipode, delta_k};
use std::collections::HashMap;
use num_rational::Rational64;
use serde::{Serialize, Deserialize};

/// A general element in the Hopf algebra as a linear combination of forests
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HopfElement {
    /// Coefficients for each forest
    terms: HashMap<Forest, Rational64>,
}

impl HopfElement {
    /// Create a zero element
    pub fn zero() -> Self {
        HopfElement {
            terms: HashMap::new(),
        }
    }

    /// Create a unit element (empty forest with coefficient 1)
    pub fn one() -> Self {
        let mut terms = HashMap::new();
        terms.insert(Forest::empty(), Rational64::from(1));
        HopfElement { terms }
    }

    /// Create from a single tree
    pub fn from_tree(tree: Tree) -> Self {
        let mut terms = HashMap::new();
        terms.insert(Forest::single(tree), Rational64::from(1));
        HopfElement { terms }
    }

    /// Create from a forest
    pub fn from_forest(forest: Forest) -> Self {
        let mut terms = HashMap::new();
        terms.insert(forest, Rational64::from(1));
        HopfElement { terms }
    }

    /// Add two elements
    pub fn add(&self, other: &HopfElement) -> HopfElement {
        let mut terms = self.terms.clone();
        
        for (forest, coeff) in &other.terms {
            *terms.entry(forest.clone()).or_insert(Rational64::from(0)) += coeff;
        }
        
        // Remove zero terms
        terms.retain(|_, coeff| *coeff != Rational64::from(0));
        
        HopfElement { terms }
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: Rational64) -> HopfElement {
        let mut terms = HashMap::new();
        
        for (forest, coeff) in &self.terms {
            let new_coeff = coeff * scalar;
            if new_coeff != Rational64::from(0) {
                terms.insert(forest.clone(), new_coeff);
            }
        }
        
        HopfElement { terms }
    }

    /// Multiply two elements (using forest multiplication)
    pub fn multiply(&self, other: &HopfElement) -> HopfElement {
        let mut terms = HashMap::new();
        
        for (f1, c1) in &self.terms {
            for (f2, c2) in &other.terms {
                let combined = f1.multiply(f2);
                let coeff = c1 * c2;
                *terms.entry(combined).or_insert(Rational64::from(0)) += coeff;
            }
        }
        
        terms.retain(|_, coeff| *coeff != Rational64::from(0));
        HopfElement { terms }
    }

    /// Get coefficient of a forest
    pub fn coefficient(&self, forest: &Forest) -> Rational64 {
        self.terms.get(forest).cloned().unwrap_or(Rational64::from(0))
    }

    /// Check if element is zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get all non-zero terms
    pub fn terms(&self) -> &HashMap<Forest, Rational64> {
        &self.terms
    }
}

/// The Hopf algebra of rooted trees
pub struct HopfAlgebra;

impl HopfAlgebra {
    /// Compute coproduct of a Hopf element
    pub fn coproduct(element: &HopfElement) -> HashMap<(HopfElement, HopfElement), Rational64> {
        let mut result = HashMap::new();
        
        for (forest, coeff) in element.terms() {
            // Coproduct is multiplicative on forests
            let mut forest_cop = HashMap::new();
            forest_cop.insert((Forest::empty(), forest.clone()), 1);
            
            for tree in forest.iter() {
                let tree_cop = tree.coproduct();
                let mut next_cop = HashMap::new();
                
                for ((left, right), c1) in forest_cop {
                    for ((tree_left, tree_right), c2) in &tree_cop {
                        let new_left = left.multiply(tree_left);
                        let new_right = right.multiply(&Forest::single(tree_right.clone()));
                        
                        *next_cop.entry((new_left, new_right)).or_insert(0) += c1 * c2;
                    }
                }
                
                forest_cop = next_cop;
            }
            
            // Convert to HopfElements
            for ((left, right), c) in forest_cop {
                let left_elem = HopfElement::from_forest(left);
                let right_elem = HopfElement::from_forest(right);
                *result.entry((left_elem, right_elem)).or_insert(Rational64::from(0)) 
                    += Rational64::from(c) * coeff;
            }
        }
        
        result
    }

    /// Compute antipode of a Hopf element
    pub fn antipode(element: &HopfElement) -> HopfElement {
        let mut result = HopfElement::zero();
        
        for (forest, coeff) in element.terms() {
            let s_forest = forest.antipode();
            let s_elem = HopfElement::from_forest(s_forest);
            result = result.add(&s_elem.scale(*coeff));
        }
        
        result
    }

    /// Generate δ_k as a Hopf element
    pub fn delta(k: usize) -> HopfElement {
        let trees = delta_k(k);
        let mut terms = HashMap::new();
        
        for tree in trees {
            let forest = Forest::single(tree);
            *terms.entry(forest).or_insert(Rational64::from(0)) += 1;
        }
        
        HopfElement { terms }
    }

    /// Verify the antipode axiom: (id ⊗ S) ∘ Δ = unit ∘ ε
    pub fn verify_antipode_axiom(tree: &Tree) -> bool {
        let cop = tree.coproduct();
        let mut sum = HopfElement::zero();
        
        for ((forest, tree), coeff) in cop {
            let left = HopfElement::from_forest(forest);
            let right = HopfElement::from_tree(tree);
            let s_right = Self::antipode(&right);
            
            let term = left.multiply(&s_right).scale(Rational64::from(coeff));
            sum = sum.add(&term);
        }
        
        // Check if sum equals ε(tree) * 1
        let expected = if tree.size() == 0 {
            HopfElement::one()
        } else {
            HopfElement::zero()
        };
        
        sum == expected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hopf_element_arithmetic() {
        let t1 = Tree::new();
        let h1 = HopfElement::from_tree(t1.clone());
        let h2 = HopfElement::from_tree(t1);
        
        let sum = h1.add(&h2);
        assert_eq!(sum.coefficient(&Forest::single(Tree::new())), Rational64::from(2));
    }

    #[test]
    fn test_delta_generation() {
        let d1 = HopfAlgebra::delta(1);
        assert_eq!(d1.terms().len(), 1); // One tree of size 1
        
        let d2 = HopfAlgebra::delta(2);
        assert_eq!(d2.terms().len(), 1); // One tree of size 2
        
        let d3 = HopfAlgebra::delta(3);
        assert_eq!(d3.terms().len(), 2); // Two trees of size 3
    }
} 