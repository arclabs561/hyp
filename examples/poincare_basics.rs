//! Poincare Ball Basics
//!
//! Demonstrates basic operations in hyperbolic space using the Poincare ball model.
//!
//! ```bash
//! cargo run --example poincare_basics --release
//! ```

use hyp::PoincareBall;
use ndarray::array;

fn main() {
    println!("Poincare Ball Basics");
    println!("====================\n");

    let ball = PoincareBall::<f64>::new(1.0); // curvature c=1

    // 1. Basic operations
    println!("1. Basic Operations");
    let origin = array![0.0, 0.0];
    let x = array![0.3, 0.0];
    let y = array![0.0, 0.3];

    println!("   Origin: {:?}", origin.to_vec());
    println!("   x = {:?}", x.to_vec());
    println!("   y = {:?}", y.to_vec());

    let d_xy = ball.distance(&x.view(), &y.view());
    let d_ox = ball.distance(&origin.view(), &x.view());
    let d_oy = ball.distance(&origin.view(), &y.view());

    println!("\n   Distances:");
    println!("     d(origin, x) = {:.4}", d_ox);
    println!("     d(origin, y) = {:.4}", d_oy);
    println!("     d(x, y)      = {:.4}", d_xy);

    // 2. Mobius addition (hyperbolic translation)
    println!("\n2. Mobius Addition (Hyperbolic Translation)");
    let sum = ball.mobius_add(&x.view(), &y.view());
    println!("   x + y (Mobius) = {:?}", sum.to_vec());
    println!("   Note: This is NOT vector addition - it follows the hyperbolic metric");

    // 3. Distance growth near boundary
    println!("\n3. Distance Growth Near Boundary");
    println!("   Hyperbolic space has distances that grow rapidly near the boundary.");
    println!("   Points that look close in Euclidean terms are far in hyperbolic distance.\n");

    let radii = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99];
    println!("   Euclidean norm | Hyperbolic distance from origin");
    println!("   --------------|----------------------------------");
    for r in radii {
        let point = array![r, 0.0];
        let d: f64 = ball.distance(&origin.view(), &point.view());
        println!("   {:.2}           | {:.4}", r, d);
    }
    println!("\n   Notice how distance accelerates near the boundary (r → 1)!");

    // 4. Exp/Log maps
    println!("\n4. Exponential and Logarithmic Maps");
    println!("   These convert between tangent space (Euclidean) and manifold (hyperbolic).");

    let tangent_vec = array![0.5, 0.3];
    println!("   Tangent vector v = {:?}", tangent_vec.to_vec());

    let on_manifold = ball.exp_map_zero(&tangent_vec.view());
    println!("   exp_0(v) = {:?}", on_manifold.to_vec());
    let on_norm: f64 = on_manifold.dot(&on_manifold);
    println!(
        "   ||exp_0(v)|| = {:.4} (always < 1 due to tanh)",
        on_norm.sqrt()
    );

    let recovered = ball.log_map_zero(&on_manifold.view());
    println!(
        "   log_0(exp_0(v)) = {:?} (should recover v)",
        recovered.to_vec()
    );

    // 5. Projection
    println!("\n5. Projection onto Ball");
    let outside = array![2.0_f64, 1.5];
    println!("   Point outside ball: {:?}", outside.to_vec());
    println!("   ||outside|| = {:.4}", {
        let norm_sq: f64 = outside.dot(&outside);
        norm_sq.sqrt()
    });

    let projected = ball.project(&outside.view());
    println!("   Projected: {:?}", projected.to_vec());
    println!(
        "   ||projected|| = {:.4} (just inside boundary)",
        projected.dot(&projected).sqrt()
    );

    // 6. Curvature effects
    println!("\n6. Curvature Effects");
    println!("   Higher curvature = smaller ball radius = faster distance growth");

    let test_point = array![0.4, 0.0];
    for c in [0.5f64, 1.0, 2.0, 4.0] {
        let ball_c = PoincareBall::<f64>::new(c);
        let d: f64 = ball_c.distance(&origin.view(), &test_point.view());
        let max_norm = (1.0 / c).sqrt();
        println!(
            "   c={:.1}: ball radius={:.3}, d(0, [0.4,0])={:.4}",
            c, max_norm, d
        );
    }

    println!("\nDone!");
}
