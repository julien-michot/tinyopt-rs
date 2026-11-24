// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Minimum-degree fill-reducing ordering (AMD family).
//!
//! The ordering is computed on the sparsity pattern of a symmetric matrix with
//! a sparse adjacency graph: repeatedly eliminate the vertex of smallest degree,
//! connecting its remaining neighbours (the elimination clique). This is the
//! classical exact minimum-degree strategy used by sparse LDLᵀ solvers; it
//! reduces fill-in compared with the natural ordering on irregular graphs.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Compute a minimum-degree permutation.
///
/// Returns `perm` where `perm[k]` is the original index eliminated at step `k`,
/// so the permuted matrix is `A(perm, perm)`.
pub fn minimum_degree(n: usize, col_ptrs: &[usize], row_indices: &[usize]) -> Vec<usize> {
    // Degrees can be counted directly from the pattern: for a lower-triangular
    // input every undirected edge contributes one endpoint to each side.
    let mut degree = vec![0usize; n];
    for j in 0..n {
        for &i in &row_indices[col_ptrs[j]..col_ptrs[j + 1]] {
            if i != j {
                degree[i] += 1;
                degree[j] += 1;
            }
        }
    }
    // Paths and cycles (maximum degree 2) are already fill-free under any
    // elimination order, so skip the ordering work entirely.
    if degree.iter().all(|&d| d <= 2) {
        return (0..n).collect();
    }

    // Sorted, deduplicated symmetric adjacency lists.
    let mut adj: Vec<Vec<usize>> = degree.iter().map(|&d| Vec::with_capacity(d)).collect();
    for j in 0..n {
        for &i in &row_indices[col_ptrs[j]..col_ptrs[j + 1]] {
            if i != j {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    for list in adj.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }

    let mut eliminated = vec![false; n];
    let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    for (i, list) in adj.iter().enumerate() {
        heap.push(Reverse((list.len(), i)));
    }

    let mut perm = Vec::with_capacity(n);
    while perm.len() < n {
        // Pop the current minimum-degree vertex (lazy deletion of stale entries).
        let node = loop {
            match heap.pop() {
                Some(Reverse((d, v))) if !eliminated[v] && d == adj[v].len() => break v,
                Some(_) => continue,
                None => {
                    // Fallback (should not happen): linear scan.
                    let mut best = usize::MAX;
                    let mut best_degree = usize::MAX;
                    for i in 0..n {
                        if !eliminated[i] && adj[i].len() < best_degree {
                            best_degree = adj[i].len();
                            best = i;
                        }
                    }
                    break best;
                }
            }
        };

        // Eliminate `node`: its remaining neighbours become a clique.
        let neighbors = std::mem::take(&mut adj[node]);
        for (a, &u) in neighbors.iter().enumerate() {
            for &v in &neighbors[a + 1..] {
                if let Err(pos) = adj[u].binary_search(&v) {
                    adj[u].insert(pos, v);
                }
                if let Err(pos) = adj[v].binary_search(&u) {
                    adj[v].insert(pos, u);
                }
            }
        }
        for &v in &neighbors {
            if let Ok(pos) = adj[v].binary_search(&node) {
                adj[v].remove(pos);
                heap.push(Reverse((adj[v].len(), v)));
            }
        }
        eliminated[node] = true;
        perm.push(node);
    }

    perm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::SparseMatrix;

    fn adjacency_of(m: &SparseMatrix<f64>) -> (Vec<usize>, Vec<usize>) {
        let lower = m.to_lower();
        (lower.col_ptrs().to_vec(), lower.row_indices().to_vec())
    }

    #[test]
    fn star_eliminates_leaves_first() {
        // Star graph: node 0 connected to every leaf.
        let n = 6;
        let mut triplets = Vec::new();
        for i in 1..n {
            triplets.push((i, 0, 1.0));
            triplets.push((0, i, 1.0));
        }
        let m = SparseMatrix::from_triplets(n, n, &triplets);
        let (ptr, idx) = adjacency_of(&m);
        let perm = minimum_degree(n, &ptr, &idx);
        // Minimum degree starts with a leaf (degree 1), never the centre.
        assert_ne!(perm[0], 0);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn chain_keeps_the_natural_order() {
        // A chain has degree <= 2 everywhere, so no reordering is needed.
        let n = 8;
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.0));
            if i + 1 < n {
                triplets.push((i + 1, i, 1.0));
                triplets.push((i, i + 1, 1.0));
            }
        }
        let m = SparseMatrix::from_triplets(n, n, &triplets);
        let (ptr, idx) = adjacency_of(&m);
        assert_eq!(minimum_degree(n, &ptr, &idx), (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn diagonal_matrix_is_any_permutation() {
        let n = 4;
        let m = SparseMatrix::from_triplets(n, n, &(0..n).map(|i| (i, i, 1.0)).collect::<Vec<_>>());
        let (ptr, idx) = adjacency_of(&m);
        let perm = minimum_degree(n, &ptr, &idx);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }
}
