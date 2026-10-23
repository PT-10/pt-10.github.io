> My write-up of the Jane Street puzzle [I dropped a neural net](https://huggingface.co/spaces/jane-street/droppedaneuralnet). The code lives on [GitHub ↗](https://github.com/PT-10/droppedaneuralnet).

## The Problem

A trained model was "dropped" and fell apart into its individual linear layers. All that's left is 97 shuffled weight pieces and some historical data: 10,000 inputs, the original model's predictions (`pred`) and the ground truth (`true`).

The goal: recover the exact original permutation (for each position 0–96, which piece index goes there) and submit it as a comma-separated list.

## The Model Architecture

```text
Input x in R^48
  Block 0:   inp: Linear(48->96) -> ReLU -> out: Linear(96->48)  [residual]
  Block 1:   same
  ...
  Block 47:  same
  Final:     Linear(48->1)
```

Each block is a residual unit:

```text
x_new = x + W_out @ ReLU(W_in @ x)
```

The residual connection means each block learns a *correction* to the identity. That property is central to every reconstruction heuristic below.

There are 48 blocks × 2 layers + 1 final layer = 97 weight matrices, matching the 97 pieces exactly.

| Shape | Role | Count |
| --- | --- | --- |
| `[96, 48]` | `block_in` | 48 |
| `[48, 96]` | `block_out` | 48 |
| `[1, 48]` | `final` | 1 |

## Step 1 - Identify Piece Types

Tensor shape is the only unambiguous label we have. All 97 pieces are loaded and sorted into three buckets by their weight shapes. The `[1, 48]` final layer is unique, so it's identified trivially.

## Step 2 - Frobenius Norm Sort

For each piece, compute the Frobenius norm:

$$
\|W\|_F = \sqrt{\sum_{i,j} w_{ij}^2}
$$

Both the `block_in` and `block_out` lists are sorted by **ascending** Frobenius norm (low norm = shallow, high norm = deep).

**Why ascending and not descending?** This was discovered empirically. The descending sort gave MSE = 449; the ascending sort gave MSE = 0.880. The skewness of the extreme blocks confirmed it after the fact:

```text
Highest-norm (current block 0):  skew ~= 0.006, -0.020   (near symmetric = shallow)
Lowest-norm  (current block 47): skew ~= 0.249, -0.327   (asymmetric = deep/specialised)
```

Higher skewness means more specialised, which means deeper in the network. There is no universal rule here; I tested both directions and the data decided.

```text
MSE: 0.880  (down from 449)
```

## Step 3 - Hungarian Algorithm: Optimal Within-Block Pairing

The norm sort gives a depth order, but `block_in` and `block_out` were sorted independently. Block 0's `inp` layer isn't necessarily paired with block 0's `out` layer. All 96 matrices were scrambled, so we need to find which 48 `block_in` pieces belong with which 48 `block_out` pieces.

This is a **bipartite matching problem**: find the optimal one-to-one assignment of 48 out-pieces to 48 in-pieces.

### Scoring a candidate pair (W_out, W_in)

In a trained residual block, `W_in` and `W_out` co-adapt during training:

- `W_in` learns which directions in the 48-dim input space to project into the 96-dim hidden space.
- `W_out` learns which of those 96 hidden units to read back out and add to the residual.

A true pair leaves algebraic signatures:

**Metric 1 - Trace of |W_out @ W_in|.** The product is a 48×48 matrix. For a true pair, the two layers co-adapted to work along the same directions, so the diagonal dominance of the product is higher than for a random mismatch.

**Metric 2 - Negative effective rank of W_out @ W_in.**

$$
\text{eff\_rank} = \exp\left(-\sum_i \hat\sigma_i \log \hat\sigma_i\right), \qquad \hat\sigma_i = \frac{\sigma_i}{\sum_j \sigma_j}
$$

A true pair produces a *low* effective rank product because the layers compressed features together along a few directions. A mismatched pair scatters energy across many dimensions. Negating it makes higher = better.

**Metric 3 - Bias-activation column variance.** The pre-ReLU bias of `W_in` decides which of the 96 hidden units fire. A true `W_out` has learned to read from exactly those active units, so its columns at those indices have higher variance.

Each metric is normalised to [0, 1] and summed. The **Hungarian algorithm** then finds the globally optimal one-to-one assignment in $O(n^3)$: exact, not approximate.

**Result:** 44 of 48 blocks were re-paired compared with the naive norm-rank pairing.

```text
MSE: 0.177  (down from 0.880)
```

## Step 4 - Coordinate Descent: Optimize Against `pred`

After Hungarian pairing, MSE against `true` is 0.177: better than 0.880, but not the exact permutation. The remaining error is in the block *sequence*.

The `pred` column is the original model's exact output on the 10,000 inputs. The correct permutation must reproduce `pred` with MSE = 0, and that's the objective the puzzle really sets.

### The algorithm

Greedy coordinate descent:

1. Try all 48 × 47 / 2 = 1,128 pairwise block position swaps.
2. Accept a swap immediately if it reduces MSE against `pred`.
3. Repeat until no swap improves it.
4. Exit early as soon as MSE against `pred` hits 0.0.

The warm start from steps 2 and 3 puts us in the basin of the correct solution, so greedy descent converges without getting trapped in a bad local minimum.

```text
Start (after Hungarian):  MSE_pred=0.076     MSE_true=0.177
Pass 1:                   MSE_pred=0.024     MSE_true=0.121
Pass 2:                   MSE_pred=0.023     MSE_true=0.121
Pass 3-4:                 MSE_pred=0.000060
Pass 5:                   MSE_pred=0.000000  MSE_true=0.106481  (early exit)
```

MSE_true = 0.106481 exactly matches the original model's baseline accuracy. Permutation fully recovered.

## Results Summary

| Stage | Method | MSE_pred | MSE_true |
| --- | --- | --- | --- |
| Norm-sorted descending | Wrong depth direction | - | 449.01 |
| Norm-sorted ascending | Correct depth direction | - | 0.880 |
| + Hungarian pairing | Optimal inp/out matching | 0.076 | 0.177 |
| **+ Coordinate descent (vs pred)** | **Block sequence recovery** | **0.000** | **0.106** |

Total runtime: about 10 minutes on a CPU.

## What Didn't Work

### SVD Sequence Reordering

**Idea:** for consecutive blocks A → B, block B's `W_in` row space should align with block A's `W_out` column space. Score every ordered pair by SVD overlap:

$$
\text{score}(A \to B) = \left\| \operatorname{diag}(S_A)\, U_A^\top V_B\, \operatorname{diag}(S_B) \right\|_F
$$

then find the best block sequence with greedy nearest-neighbour plus 2-opt local search over this alignment graph.

**Result:** MSE 0.875.

### Adjacent-Only Mid-Layer Swaps

**Idea:** target only the mid-layer blocks (indices 16–31) with the lowest combined scores, try swapping them with their immediate neighbours, then do a left-to-right adjacent scan.

**Result:** a modest MSE improvement, but it never converged to the solution.

**Why it didn't finish the job:** adjacent swaps are too conservative. The biggest gains in the full coordinate descent came from non-local swaps in the deep layers (blocks 38–47 moved many positions). Limiting swaps to ±1 in the mid-range missed the actual disorder entirely.

## Final Solution

```text
43, 34, 65, 22, 69, 89, 28, 12, 27, 76, 81, 8, 5, 21, 62, 79,
64, 70, 94, 96, 4, 17, 48, 9, 23, 46, 14, 33, 95, 26, 50, 66,
1, 40, 15, 67, 41, 92, 16, 83, 77, 32, 10, 20, 3, 53, 45, 19,
87, 71, 88, 54, 39, 38, 18, 25, 56, 30, 91, 29, 44, 82, 35, 24,
61, 80, 86, 57, 31, 36, 13, 7, 59, 52, 68, 47, 84, 63, 74,
90, 0, 75, 73, 11, 37, 6, 58, 78, 42, 55, 49, 72, 2, 51, 60,
93, 85
```

![The puzzle page confirming the reconstruction](dropped-a-neural-net/solved.png)
