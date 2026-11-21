// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! C++ tinyopt backend: runs the Catch2 benchmark binary and parses its output.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use quick_xml::Reader;
use quick_xml::events::Event;

/// Parsed results of one C++ benchmark run.
#[derive(Default)]
pub struct CppResults {
    /// Benchmark label -> mean runtime in nanoseconds.
    pub means: BTreeMap<String, f64>,
    /// (category, problem) -> (iterations, converged).
    pub iterations: BTreeMap<(String, String), (u32, bool)>,
}

/// Run the C++ benchmark binary; returns `None` when it is unavailable.
pub fn run(binary: &Path, samples: usize, warmup_seconds: f64) -> Option<CppResults> {
    if !binary.is_file() {
        return None;
    }
    let output = Command::new(binary)
        .arg("[benchmark]")
        .arg("--benchmark-no-analysis")
        .arg("--benchmark-samples")
        .arg(samples.to_string())
        .arg("--benchmark-warmup-time")
        .arg(warmup_seconds.to_string())
        .arg("--reporter")
        .arg("xml")
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("VECLIB_MAXIMUM_THREADS", "1")
        .env("NUMEXPR_NUM_THREADS", "1")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let mut results = CppResults::default();
    for line in stdout.lines().chain(stderr.lines()) {
        let Some(rest) = line.strip_prefix("[iterations] ") else {
            continue;
        };
        let parts: Vec<&str> = rest.split('|').map(str::trim).collect();
        if parts.len() != 5 || parts[2] != "tinyopt" {
            continue;
        }
        let Ok(count) = parts[3].parse::<u32>() else {
            continue;
        };
        results.iterations.insert(
            (parts[0].to_string(), parts[1].to_string()),
            (count, parts[4] == "ok"),
        );
    }

    let start = stdout.find("<Catch2TestRun")?;
    let end = stdout.find("</Catch2TestRun>")?;
    let xml = &stdout[start..end + "</Catch2TestRun>".len()];
    parse_xml(xml, &mut results.means);
    Some(results)
}

fn parse_xml(xml: &str, means: &mut BTreeMap<String, f64>) {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut test_case = String::new();
    let mut benchmark = String::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(event)) | Ok(Event::Empty(event)) => {
                let name = event.name();
                match name.as_ref() {
                    b"TestCase" => {
                        test_case = attribute(&event, b"name").unwrap_or_default();
                    }
                    b"BenchmarkResults" => {
                        benchmark = attribute(&event, b"name").unwrap_or_default();
                    }
                    b"mean" => {
                        if let Some(value) =
                            attribute(&event, b"value").and_then(|v| v.parse::<f64>().ok())
                        {
                            let label = if benchmark.is_empty() {
                                test_case.clone()
                            } else {
                                benchmark.clone()
                            };
                            means.insert(label, value);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
}

fn attribute(event: &quick_xml::events::BytesStart<'_>, key: &[u8]) -> Option<String> {
    event
        .attributes()
        .flatten()
        .find(|attribute| attribute.key.as_ref() == key)
        .and_then(|attribute| String::from_utf8(attribute.value.into_owned()).ok())
}
