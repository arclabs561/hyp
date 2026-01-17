//! Hyperbolic geometry for embedding hierarchical structures.
//!
//! Embed trees and hierarchies in low dimensions where Euclidean would need thousands.
//!
//! # Which Model Should I Use?
//!
//! | Task | Model | Why |
//! |------|-------|-----|
//! | **Learning embeddings** | [`LorentzModel`] | Stable gradients everywhere |
//! | **Visualization** | [`PoincareBall`] | Bounded, intuitive |
//! | **Mixed hierarchy + similarity** | Consider mixed-curvature | Best of both |
//!
//! # Why Hyperbolic?
//!
//! | Year | Observation | Implication |
//! |------|-------------|-------------|
//! | 2017 | Trees embed poorly in Euclidean space | Need exponential dims |
//! | 2017 | Hyperbolic space has exponential volume | Trees embed naturally |
//! | 2018 | Lorentz model more stable | Better for optimization |
//! | 2021 | Mixed-curvature spaces | Best of both worlds |
//!
//! **Key insight**: Hyperbolic space has *exponentially growing* volume
//! with radius, just like trees have exponentially growing nodes with depth.
//! A 10-dimensional hyperbolic space can embed trees that would require
//! thousands of Euclidean dimensions.
//!
//! # The Two Models
//!
//! | Model | Representation | Pros | Cons |
//! |-------|---------------|------|------|
//! | Poincaré Ball | Unit ball | Intuitive, conformal | Gradients vanish at boundary |
//! | Lorentz (Hyperboloid) | Upper sheet of hyperboloid | Stable gradients | Less intuitive |
//!
//! For **learning**, prefer Lorentz: gradients are well-behaved everywhere.
//! For **visualization**, prefer Poincaré: it's a bounded disk.
//!
//! # Mathematical Background
//!
//! The Poincaré ball model represents hyperbolic space as the interior of
//! a unit ball with the metric:
//!
//! ```text
//! ds² = (2/(1-||x||²))² ||dx||²
//! ```
//!
//! As points approach the boundary (||x|| → 1), distances grow infinitely—
//! this is how infinite hierarchical depth fits in finite Euclidean volume.
//!
//! # When to Use
//!
//! - **Taxonomies**: WordNet, Wikipedia categories
//! - **Organizational hierarchies**: Company structures, file systems
//! - **Evolutionary trees**: Phylogenetics, language families
//! - **Social networks**: Often have hierarchical community structure
//!
//! # When NOT to Use
//!
//! - **Flat structures**: No hierarchy to exploit (use Euclidean)
//! - **Grid-like data**: Images, audio (use CNN/RNN)
//! - **Very shallow trees**: Depth < 5, Euclidean often suffices
//!
//! # Connection to Intrinsic Dimension
//!
//! Local Intrinsic Dimensionality (LID) can help decide between hyperbolic
//! and Euclidean embeddings:
//!
//! - **Low LID + hierarchical structure**: Use hyperbolic (Poincaré/Lorentz)
//! - **High LID + uniform structure**: Use Euclidean (HNSW, IVF-PQ)
//! - **Variable LID across regions**: Consider mixed-curvature spaces
//!
//! Research (D-Mercator, 2023) shows that networks with low intrinsic
//! dimension in hyperbolic space exhibit high navigability—meaning greedy
//! routing succeeds. This connects to HNSW's small-world navigation: graphs
//! with low effective dimension are easier to search.
//!
//! See `jin::lid` for LID estimation utilities.
//!
//! # References
//!
//! - Nickel & Kiela (2017): "Poincaré Embeddings for Learning Hierarchical Representations"
//! - Nickel & Kiela (2018): "Learning Continuous Hierarchies in the Lorentz Model"
//! - Chami et al. (2019): "Hyperbolic Graph Convolutional Neural Networks"

use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive, Zero};

pub mod lorentz;

pub use lorentz::LorentzModel;

/// Poincaré Ball manifold.
///
/// The Poincaré ball is the interior of the unit ball {x : ||x|| < 1/√c}
/// equipped with the hyperbolic metric. Curvature c controls the "strength"
/// of hyperbolic effects.
pub struct PoincareBall<T> {
    /// Curvature parameter (c > 0)
    pub c: T,
}

impl<T> PoincareBall<T>
where
    T: Float + FromPrimitive + Zero + ndarray::ScalarOperand + ndarray::LinalgScalar,
{
    pub fn new(c: T) -> Self {
        Self { c }
    }

    /// Mobius addition: x + y = (1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y / (1 + 2c<x,y> + c^2||x||^2||y||^2)
    pub fn mobius_add(&self, x: &ArrayView1<T>, y: &ArrayView1<T>) -> Array1<T> {
        let x_norm_sq = x.dot(x);
        let y_norm_sq = y.dot(y);
        let xy = x.dot(y);

        let c = self.c;
        let one = T::one();
        let two = T::from_f64(2.0).unwrap();

        let denom = one + two * c * xy + c * c * x_norm_sq * y_norm_sq;

        let term1 = (one + two * c * xy + c * y_norm_sq) * x;
        let term2 = (one - c * x_norm_sq) * y;

        (term1 + term2) / denom
    }

    /// Hyperbolic distance.
    pub fn distance(&self, x: &ArrayView1<T>, y: &ArrayView1<T>) -> T {
        let neg_x = -x;
        let diff = self.mobius_add(&neg_x.view(), y);
        let diff_norm = diff.dot(&diff).sqrt();
        let c_sqrt = self.c.sqrt();
        let two = T::from_f64(2.0).unwrap();

        two / c_sqrt * (c_sqrt * diff_norm).atanh()
    }

    /// Logarithmic map at origin (tangent space -> manifold).
    /// For origin, log_0(y) = y.
    pub fn log_map_zero(&self, y: &ArrayView1<T>) -> Array1<T> {
        let y_norm = y.dot(y).sqrt();
        let epsilon = T::from_f64(1e-7).unwrap(); // f32 friendly epsilon
        
        if y_norm < epsilon {
            return y.to_owned();
        }
        let c_sqrt = self.c.sqrt();
        let scale = (c_sqrt * y_norm).atanh() / (c_sqrt * y_norm);
        y * scale
    }

    /// Exponential map at origin (manifold -> tangent space).
    /// exp_0(v) = tanh(sqrt(c)*||v||) / (sqrt(c)*||v||) * v
    pub fn exp_map_zero(&self, v: &ArrayView1<T>) -> Array1<T> {
        let v_norm = v.dot(v).sqrt();
        let epsilon = T::from_f64(1e-7).unwrap();

        if v_norm < epsilon {
            return v.to_owned();
        }
        let c_sqrt = self.c.sqrt();
        let scale = (c_sqrt * v_norm).tanh() / (c_sqrt * v_norm);
        v * scale
    }

    /// Check if point is inside the Poincare ball (||x|| < 1/sqrt(c)).
    pub fn is_in_ball(&self, x: &ArrayView1<T>) -> bool {
        let norm_sq = x.dot(x);
        norm_sq < T::one() / self.c
    }

    /// Project point onto ball boundary if outside.
    pub fn project(&self, x: &ArrayView1<T>) -> Array1<T> {
        let norm = x.dot(x).sqrt();
        let one = T::one();
        let epsilon = T::from_f64(1e-5).unwrap();
        let max_norm = (one / self.c).sqrt() - epsilon;
        
        if norm > max_norm {
            x * (max_norm / norm)
        } else {
            x.to_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_distance_self_is_zero() {
        let ball = PoincareBall::new(1.0);
        let x = array![0.1, 0.2, 0.3];
        let d = ball.distance(&x.view(), &x.view());
        assert!(d.abs() < EPS, "distance to self should be 0, got {}", d);
    }

    #[test]
    fn test_distance_symmetric() {
        let ball = PoincareBall::new(1.0);
        let x = array![0.1, 0.2];
        let y = array![0.3, -0.1];
        let d_xy = ball.distance(&x.view(), &y.view());
        let d_yx = ball.distance(&y.view(), &x.view());
        assert!((d_xy - d_yx).abs() < EPS, "distance not symmetric");
    }

    #[test]
    fn test_distance_non_negative() {
        let ball = PoincareBall::new(1.0);
        let x = array![0.1, 0.2];
        let y = array![0.3, -0.1];
        let d = ball.distance(&x.view(), &y.view());
        assert!(d >= 0.0, "distance should be non-negative");
    }

    #[test]
    fn test_distance_triangle_inequality() {
        let ball = PoincareBall::new(1.0);
        let a = array![0.1, 0.0];
        let b = array![0.0, 0.1];
        let c = array![-0.1, 0.0];

        let d_ac = ball.distance(&a.view(), &c.view());
        let d_ab = ball.distance(&a.view(), &b.view());
        let d_bc = ball.distance(&b.view(), &c.view());

        assert!(
            d_ac <= d_ab + d_bc + EPS,
            "triangle inequality violated: {} > {} + {}",
            d_ac,
            d_ab,
            d_bc
        );
    }

    #[test]
    fn test_mobius_add_identity() {
        let ball = PoincareBall::new(1.0);
        let x = array![0.1, 0.2, 0.3];
        let zero = array![0.0, 0.0, 0.0];

        // x + 0 = x
        let result = ball.mobius_add(&x.view(), &zero.view());
        for i in 0..3 {
            assert!(
                (result[i] - x[i]).abs() < EPS,
                "mobius_add with zero failed at index {}",
                i
            );
        }
    }

    #[test]
    fn test_exp_log_round_trip() {
        let ball = PoincareBall::new(1.0);
        let v = array![0.3, 0.2, 0.1]; // tangent vector

        // exp then log should recover original (approximately)
        let on_manifold = ball.exp_map_zero(&v.view());
        let recovered = ball.log_map_zero(&on_manifold.view());

        for i in 0..3 {
            assert!(
                (recovered[i] - v[i]).abs() < 1e-6,
                "exp/log round trip failed at index {}: {} vs {}",
                i,
                recovered[i],
                v[i]
            );
        }
    }

    #[test]
    fn test_exp_map_stays_in_ball() {
        let ball = PoincareBall::new(1.0);

        // Even large tangent vectors should map to inside the ball
        let large_v = array![10.0, 10.0, 10.0];
        let result = ball.exp_map_zero(&large_v.view());

        assert!(
            ball.is_in_ball(&result.view()),
            "exp_map result escaped the ball"
        );
    }

    #[test]
    fn test_project_inside_unchanged() {
        let ball = PoincareBall::new(1.0);
        let inside = array![0.1, 0.2]; // clearly inside unit ball
        let projected = ball.project(&inside.view());

        for i in 0..2 {
            assert!(
                (projected[i] - inside[i]).abs() < EPS,
                "projection changed point already inside ball"
            );
        }
    }

    #[test]
    fn test_project_outside_onto_boundary() {
        let ball = PoincareBall::new(1.0);
        let outside = array![2.0, 0.0]; // clearly outside unit ball
        let projected = ball.project(&outside.view());

        assert!(
            ball.is_in_ball(&projected.view()),
            "projection did not bring point inside ball"
        );
    }

    #[test]
    fn test_curvature_affects_distance() {
        // Different curvatures should give different distances
        // Note: the relationship is subtle - higher c means smaller ball radius
        let ball_c1 = PoincareBall::new(1.0);
        let ball_c2 = PoincareBall::new(4.0);

        let x = array![0.1, 0.0];
        let y = array![0.0, 0.1];

        let d1 = ball_c1.distance(&x.view(), &y.view());
        let d2 = ball_c2.distance(&x.view(), &y.view());

        // Just verify they're different (curvature has an effect)
        // Difference is small for points near origin but non-zero
        assert!(
            (d1 - d2).abs() > 1e-6,
            "curvature should affect distance: c=1 gives {}, c=4 gives {}",
            d1,
            d2
        );
    }
}
