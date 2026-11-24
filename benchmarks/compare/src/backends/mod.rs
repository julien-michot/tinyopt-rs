// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Backend runners.

#[cfg(feature = "ceres")]
pub mod ceres;
pub mod cpp;
pub mod tinyopt;
