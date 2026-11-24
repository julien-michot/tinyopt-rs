// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Fixed (non-optimized) parameters, mirroring `optimizer.fix(...)`.

/// Set of parameter indices held fixed during optimization.
#[derive(Clone, Debug, Default)]
pub struct FixedSet {
    indices: Vec<usize>,
}

impl FixedSet {
    /// Hold the given parameter indices fixed.
    pub fn fix(&mut self, indices: &[usize]) {
        for &i in indices {
            if !self.indices.contains(&i) {
                self.indices.push(i);
            }
        }
        self.indices.sort_unstable();
    }

    /// Release all fixed parameters.
    pub fn clear(&mut self) {
        self.indices.clear();
    }

    /// Whether any parameter is fixed.
    pub fn has_fixed(&self) -> bool {
        !self.indices.is_empty()
    }

    /// Number of fixed parameters.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Whether `index` is fixed.
    pub fn contains(&self, index: usize) -> bool {
        self.indices.binary_search(&index).is_ok()
    }

    /// Sorted fixed indices.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_set_bookkeeping() {
        let mut f = FixedSet::default();
        assert!(!f.has_fixed());
        f.fix(&[3, 1]);
        f.fix(&[1, 5]);
        assert_eq!(f.indices(), &[1, 3, 5]);
        assert_eq!(f.len(), 3);
        assert!(f.contains(3));
        assert!(!f.contains(2));
        f.clear();
        assert!(f.is_empty());
    }
}
