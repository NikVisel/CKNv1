//! Geometric embeddings and Conformal Geometric Algebra for Hopf structures

use crate::algebra::Tree;
use ndarray::{Array1, Array2};
use std::f32::consts::PI;

/// Conformal Geometric Algebra (CGA) embedding
/// Maps trees to 5D conformal space: 3D + origin + infinity
pub struct CGA {
    /// Dimension of base space (typically 3)
    base_dim: usize,
    /// Full CGA dimension (base + 2)
    cga_dim: usize,
}

impl CGA {
    /// Create new CGA embedding
    pub fn new(base_dim: usize) -> Self {
        CGA {
            base_dim,
            cga_dim: base_dim + 2,
        }
    }

    /// Embed a tree into CGA space
    pub fn embed(&self, tree: &Tree) -> CGAMultivector {
        // Map tree structure to geometric primitives
        let position = self.tree_to_position(tree);
        let radius = (tree.size() as f32).sqrt();
        
        // Create sphere in CGA
        self.create_sphere(position, radius)
    }

    /// Create a sphere in CGA representation
    fn create_sphere(&self, center: Array1<f32>, radius: f32) -> CGAMultivector {
        let mut coords = Array1::zeros(self.cga_dim);
        
        // Copy spatial coordinates
        for i in 0..self.base_dim.min(center.len()) {
            coords[i] = center[i];
        }
        
        // Conformal coordinates
        let x_squared = center.mapv(|x| x * x).sum();
        coords[self.base_dim] = (x_squared + radius * radius) / 2.0; // e_+
        coords[self.base_dim + 1] = (x_squared - radius * radius) / 2.0; // e_-
        
        CGAMultivector {
            grade: 1,
            coefficients: coords,
        }
    }

    /// Map tree structure to spatial position
    fn tree_to_position(&self, tree: &Tree) -> Array1<f32> {
        let mut pos = Array1::zeros(self.base_dim);
        
        // Use tree properties to determine position
        // This is a simple mapping - could be learned
        if self.base_dim >= 3 {
            pos[0] = tree.size() as f32;
            pos[1] = tree.max_depth() as f32;
            pos[2] = tree.leaf_count() as f32;
        }
        
        pos
    }

    /// Compute geometric product of two multivectors
    pub fn geometric_product(&self, a: &CGAMultivector, b: &CGAMultivector) -> CGAMultivector {
        // Simplified geometric product for vectors
        let mut result = Array1::zeros(self.cga_dim);
        
        // Dot product part
        let dot = a.coefficients.dot(&b.coefficients);
        
        // Wedge product part (bivector)
        // For full implementation, would need Clifford algebra rules
        
        CGAMultivector {
            grade: (a.grade + b.grade) % 2,
            coefficients: result,
        }
    }
}

/// Multivector in Conformal Geometric Algebra
#[derive(Clone, Debug)]
pub struct CGAMultivector {
    /// Grade of the multivector (0=scalar, 1=vector, 2=bivector, etc.)
    pub grade: usize,
    /// Coefficients in the CGA basis
    pub coefficients: Array1<f32>,
}

/// General geometric embedding trait
pub trait GeometricEmbedding {
    /// Embed a tree into geometric space
    fn embed(&self, tree: &Tree) -> Array1<f32>;
    
    /// Compute distance between embeddings
    fn distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32;
    
    /// Interpolate between embeddings
    fn interpolate(&self, a: &Array1<f32>, b: &Array1<f32>, t: f32) -> Array1<f32>;
}

/// Hyperbolic embedding for trees
pub struct HyperbolicEmbedding {
    /// Dimension of hyperbolic space
    dim: usize,
    /// Curvature parameter
    curvature: f32,
}

impl HyperbolicEmbedding {
    pub fn new(dim: usize, curvature: f32) -> Self {
        HyperbolicEmbedding { dim, curvature }
    }

    /// Convert to Poincaré ball coordinates
    fn to_poincare(&self, x: &Array1<f32>) -> Array1<f32> {
        let norm = x.mapv(|v| v * v).sum().sqrt();
        if norm >= 1.0 {
            x / (norm + 1e-6)
        } else {
            x.clone()
        }
    }

    /// Möbius addition in Poincaré ball
    fn mobius_add(&self, x: &Array1<f32>, y: &Array1<f32>) -> Array1<f32> {
        let xy = x.dot(y);
        let x_norm2 = x.dot(x);
        let y_norm2 = y.dot(y);
        
        let denominator = 1.0 + 2.0 * xy + x_norm2 * y_norm2;
        let numerator_x = x * (1.0 + 2.0 * xy + y_norm2);
        let numerator_y = y * (1.0 + x_norm2);
        
        (numerator_x + numerator_y) / denominator
    }
}

impl GeometricEmbedding for HyperbolicEmbedding {
    fn embed(&self, tree: &Tree) -> Array1<f32> {
        // Map tree to hyperbolic space
        let mut coords = Array1::zeros(self.dim);
        
        // Use tree structure to determine hyperbolic coordinates
        let depth = tree.max_depth() as f32;
        let size = tree.size() as f32;
        
        // Radial coordinate (distance from origin)
        let radius = (depth / (depth + 1.0)).tanh();
        
        // Angular coordinates based on tree structure
        if self.dim >= 2 {
            coords[0] = radius * (size / 10.0).sin();
            coords[1] = radius * (size / 10.0).cos();
        }
        
        for i in 2..self.dim {
            coords[i] = radius * ((i as f32) * size / 20.0).sin();
        }
        
        self.to_poincare(&coords)
    }

    fn distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        // Hyperbolic distance in Poincaré ball
        let diff = a - b;
        let a_norm2 = a.dot(a);
        let b_norm2 = b.dot(b);
        let diff_norm2 = diff.dot(&diff);
        
        let cosh_dist = 1.0 + 2.0 * diff_norm2 / ((1.0 - a_norm2) * (1.0 - b_norm2));
        cosh_dist.ln()
    }

    fn interpolate(&self, a: &Array1<f32>, b: &Array1<f32>, t: f32) -> Array1<f32> {
        // Geodesic interpolation in hyperbolic space
        if t <= 0.0 {
            return a.clone();
        }
        if t >= 1.0 {
            return b.clone();
        }
        
        // Use Möbius addition for interpolation
        let neg_a = -a;
        let v = self.mobius_add(&neg_a, b);
        let tv = v * t;
        
        self.mobius_add(a, &tv)
    }
}

/// Spherical embedding for trees
pub struct SphericalEmbedding {
    /// Dimension of the sphere (n-sphere in (n+1)-dimensional space)
    dim: usize,
}

impl SphericalEmbedding {
    pub fn new(dim: usize) -> Self {
        SphericalEmbedding { dim }
    }

    /// Project to unit sphere
    fn normalize(&self, x: &Array1<f32>) -> Array1<f32> {
        let norm = x.mapv(|v| v * v).sum().sqrt();
        if norm > 1e-6 {
            x / norm
        } else {
            let mut unit = Array1::zeros(x.len());
            unit[0] = 1.0;
            unit
        }
    }
}

impl GeometricEmbedding for SphericalEmbedding {
    fn embed(&self, tree: &Tree) -> Array1<f32> {
        let mut coords = Array1::zeros(self.dim + 1);
        
        // Map tree features to spherical coordinates
        let size = tree.size() as f32;
        let depth = tree.max_depth() as f32;
        
        // Use tree topology to determine angles
        for i in 0..=self.dim {
            let angle = (i as f32 + 1.0) * size / 10.0;
            coords[i] = if i < self.dim {
                angle.sin() * ((i + 1) as f32 * depth / 5.0).cos()
            } else {
                angle.cos()
            };
        }
        
        self.normalize(&coords)
    }

    fn distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        // Great circle distance
        let dot = a.dot(b).clamp(-1.0, 1.0);
        dot.acos()
    }

    fn interpolate(&self, a: &Array1<f32>, b: &Array1<f32>, t: f32) -> Array1<f32> {
        // Spherical linear interpolation (slerp)
        let dot = a.dot(b).clamp(-1.0, 1.0);
        let theta = dot.acos();
        
        if theta.abs() < 1e-6 {
            return a.clone();
        }
        
        let sin_theta = theta.sin();
        let a_weight = ((1.0 - t) * theta).sin() / sin_theta;
        let b_weight = (t * theta).sin() / sin_theta;
        
        self.normalize(&(a * a_weight + b * b_weight))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::TreeBuilder;

    #[test]
    fn test_cga_embedding() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1)
            .add_child(1, 2);
        let tree = builder.build().unwrap();
        
        let cga = CGA::new(3);
        let embedding = cga.embed(&tree);
        
        assert_eq!(embedding.coefficients.len(), 5); // 3D + 2 conformal
    }

    #[test]
    fn test_hyperbolic_embedding() {
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1);
        let tree = builder.build().unwrap();
        
        let hyp = HyperbolicEmbedding::new(3, -1.0);
        let embedding = hyp.embed(&tree);
        
        // Check that embedding is in Poincaré ball
        let norm = embedding.mapv(|x| x * x).sum().sqrt();
        assert!(norm < 1.0);
    }

    #[test]
    fn test_spherical_interpolation() {
        let sphere = SphericalEmbedding::new(2);
        
        let a = sphere.normalize(&Array1::from_vec(vec![1.0, 0.0, 0.0]));
        let b = sphere.normalize(&Array1::from_vec(vec![0.0, 1.0, 0.0]));
        
        let mid = sphere.interpolate(&a, &b, 0.5);
        
        // Check that interpolated point is on sphere
        let norm = mid.mapv(|x| x * x).sum().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
} 