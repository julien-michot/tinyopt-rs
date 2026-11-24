// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Build script: compile and link the optional CHOLMOD shim.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/sparse/cholmod_shim.c");
    println!("cargo:rerun-if-env-changed=CHOLMOD_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CHOLMOD_LIB_DIR");
    if std::env::var_os("CARGO_FEATURE_SUITESPARSE").is_none() {
        return;
    }

    let mut include: Option<PathBuf> = std::env::var_os("CHOLMOD_INCLUDE_DIR").map(PathBuf::from);
    let mut lib: Option<PathBuf> = std::env::var_os("CHOLMOD_LIB_DIR").map(PathBuf::from);

    if (include.is_none() || lib.is_none())
        && let Some(prefix) = std::env::var_os("CONDA_PREFIX")
    {
        let prefix = PathBuf::from(prefix);
        include.get_or_insert_with(|| prefix.join("include/suitesparse"));
        lib.get_or_insert_with(|| prefix.join("lib"));
    }
    for base in ["/opt/homebrew", "/usr/local"] {
        let root = PathBuf::from(base);
        if include.is_none() && root.join("include/suitesparse/cholmod.h").is_file() {
            include = Some(root.join("include/suitesparse"));
        }
        if lib.is_none() && root.join("lib/libcholmod.dylib").is_file() {
            lib = Some(root.join("lib"));
        }
    }
    let Some(include) = include else {
        panic!(
            "the `suitesparse` feature needs CHOLMOD: install SuiteSparse and set \
             CHOLMOD_INCLUDE_DIR/CHOLMOD_LIB_DIR (or CONDA_PREFIX)"
        );
    };
    let lib = lib.unwrap_or_else(|| PathBuf::from("/usr/lib"));

    cc::Build::new()
        .file("src/sparse/cholmod_shim.c")
        .include(&include)
        .flag_if_supported("-O3")
        .compile("tinyopt_cholmod_shim");

    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=cholmod");
    // Let the test/example binaries find libcholmod at runtime.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
}
