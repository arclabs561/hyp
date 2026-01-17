# hyp

Hyperbolic geometry primitives for representation learning in non-Euclidean spaces.
Implements the Poincare ball and Lorentz (hyperboloid) models for embedding hierarchical data.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/hyp) | [docs.rs](https://docs.rs/hyp)

```rust
use hyp::PoincareBall;
use ndarray::array;

let ball = PoincareBall::new(1.0);  // curvature c=1

let x = array![0.1, 0.2];
let y = array![0.3, -0.1];

// Hyperbolic distance
let dist = ball.distance(&x.view(), &y.view());

// Mobius addition (hyperbolic translation)
let sum = ball.mobius_add(&x.view(), &y.view());
```

## Operations

| Operation | Poincare | Lorentz |
|-----------|----------|---------|
| Distance | `distance()` | `distance()` |
| Addition | `mobius_add()` | - |
| Exp map | `exp_map_zero()` | `exp_map()` |
| Log map | `log_map_zero()` | `log_map()` |
| Project | `project()` | `project()` |

## Why Hyperbolic?

Hyperbolic space has exponentially growing volume with radius, matching how trees have exponentially growing nodes with depth. A 10-dim hyperbolic space embeds trees that would need thousands of Euclidean dimensions.

## Curvature

- `c = 1.0` — Standard hyperbolic space
- `c > 1.0` — Stronger curvature (distances grow faster)
- `c → 0` — Approaches Euclidean
