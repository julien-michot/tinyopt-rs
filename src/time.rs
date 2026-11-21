// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! High-resolution timing helpers, mirroring `tinyopt/time.h`.

/// Clock used by the library.
pub type Clock = std::time::Instant;

/// Point in time.
pub type TimePoint = std::time::Instant;

/// Current time point (`tic`).
#[inline]
pub fn tic() -> TimePoint {
    std::time::Instant::now()
}

/// Milliseconds elapsed since `t0` (`toc_ms`).
#[inline]
pub fn toc_ms(t0: TimePoint) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}

/// Milliseconds between two time points (`dt_ms`).
#[inline]
pub fn dt_ms(t0: TimePoint, t1: TimePoint) -> f64 {
    t1.duration_since(t0).as_secs_f64() * 1e3
}
