// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0
// faer stores even "static" dimensions on the heap, so this guarantee is
// specific to the nalgebra backend.
#![cfg(feature = "nalgebra")]

//! Mirrors tests/cpp/dynamic_alloc.cpp (fixed-size path): the manual
//! accumulation path must not allocate during optimization.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt::{Cost, Options, Static, optimize};

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

type V2 = Vector<f64, Static<2>>;
type M2 = Matrix<f64, Static<2>>;

fn acc() -> impl Fn(&[f64], Option<&mut V2>, Option<&mut M2>) -> Cost {
    |x: &[f64], grad: Option<&mut V2>, h: Option<&mut M2>| -> Cost {
        let (r0, r1) = (x[0] - 1.0, x[1] + 2.0);
        if let Some(g) = grad {
            g.set_at(0, 2.0 * r0);
            g.set_at(1, 2.0 * r1);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 2.0);
            h.set_elem(1, 1, 2.0);
        }
        Cost::with_num_residuals(r0 * r0 + r1 * r1, 2)
    }
}

#[test]
fn fixed_size_optimization_does_not_allocate() {
    let mut options = Options::default();
    // History would grow a Vec; the C++ strict mode uses a fixed-size history.
    options.exp.save_history = false;
    options.log.enable = false;

    // Warm up (no-op work) so that lazy initialization is excluded.
    for _ in 0..3 {
        let mut x = <V2 as DenseVector<f64>>::zeros(2);
        let summary = optimize(&mut x, acc(), &options);
        assert!(summary.succeeded());
    }

    ALLOCATIONS.store(0, Ordering::Relaxed);

    let mut x = <V2 as DenseVector<f64>>::zeros(2);
    let summary = optimize(&mut x, acc(), &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-8);
    assert_relative_eq!(x.at(1), -2.0, epsilon = 1e-8);

    let count = ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(count, 0, "fixed-size optimization allocated {count} times");
}
