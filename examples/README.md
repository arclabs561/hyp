# Examples

Hyperbolic geometry for hierarchical data.

## Quick Start

| Example | What It Teaches |
|---------|-----------------|
| `poincare_basics` | Core operations: distance, Mobius addition, exp/log maps |
| `tree_embedding` | Why 2D hyperbolic can embed trees needing O(depth) Euclidean dims |

```sh
cargo run --example poincare_basics --release
cargo run --example tree_embedding --release
```

## Real Hierarchies

| Example | What It Teaches |
|---------|-----------------|
| `taxonomy_embedding` | Embed WordNet-style taxonomy, evaluate with MAP |
| `hierarchy_recovery` | Recover parent-child structure from distances |

```sh
cargo run --example taxonomy_embedding --release
cargo run --example hierarchy_recovery --release
```

## Why Hyperbolic?

**The core insight**: Hyperbolic volume grows exponentially with radius.

```
Euclidean 2D:  area ~ r^2    (polynomial)
Hyperbolic 2D: area ~ e^r    (exponential)
```

Trees also grow exponentially: a binary tree at depth d has 2^d leaves.

**Result**: 2D hyperbolic space can embed trees with low distortion that would require O(log n) Euclidean dimensions.

## Geometric Hierarchy Stack

| Data Type | Geometry | Crate | Why |
|-----------|----------|-------|-----|
| Trees (strict) | Hyperbolic | `hyp` | Exponential volume matches tree growth |
| DAGs/Lattices | Boxes | `subsume` | Containment = entailment |
| General graphs | Euclidean | `lattix-kge` | TransE, RotatE, point embeddings |
| Dense vectors | Euclidean | `jin` | HNSW, IVF-PQ, standard ANN |

## When to Use Hyperbolic

```
Is your data a strict tree?
  └─> hyp (Poincare + Lorentz)

Is your data a DAG with multiple parents?
  └─> subsume (box embeddings)

Is your data a general graph?
  └─> lattix-kge (Euclidean KGE)

Just need nearest neighbor search?
  └─> jin (HNSW)
```
