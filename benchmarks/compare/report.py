#!/usr/bin/env python3
"""Render plots and an HTML/Markdown report from a tinyopt benchmark results.json."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import math
import re
import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BACKENDS = ("tinyopt-rs", "tinyopt-rs-cholmod", "ceres-rs", "cpp-tinyopt")
GROUP_ORDER = ("Dense static", "Dense dynamic", "Sparse")


def normalize(record: dict) -> tuple[str, str, int]:
    """Return (category, short label, order key) like the C++ report script."""
    label = str(record["problem"])
    category = str(record["category"])
    if category.startswith("Dense"):
        match = re.fullmatch(r"(\d+)D (static|dynamic) (float|double)( prior)?", label)
        if match:
            dims, storage, precision, prior = match.groups()
            short = f"{dims}{'f' if precision == 'float' else 'd'}"
            if prior:
                short += "p"
            return (f"Dense {storage}", short, int(dims))
    if "sparse chain" in label.lower():
        match = re.match(r"(\d+)D", label)
        return ("Sparse", f"{int(match.group(1))}d" if match else label, int(match.group(1)) if match else 0)
    if category == "Bundle adjustment":
        match = re.match(r"(\d+)c\s+(\d+)p(?:\s+(.*))?", label)
        if match:
            cams, points, suffix = match.groups()
            short = f"{cams}c {points}p" + (f" {suffix.strip()}" if suffix else "")
            return ("Bundle adjustment", short, int(cams) * int(points))
    if category == "Robust pose":
        match = re.match(r"(\d+)\s*obs\s*(.*)", label, re.I)
        if match:
            obs, suffix = match.groups()
            return ("Robust pose", f"{obs}o {suffix}".strip(), int(obs))
    return (category, label, 0)


def load(path: Path) -> tuple[dict, list[dict]]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for record in data["records"]:
        category, short, order = normalize(record)
        records.append({**record, "category": category, "short": short, "order": order})
    iterations = data.get("iterations", [])
    return data, records


def table_rows(records: list[dict]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for record in records:
        rows.setdefault(str(record["short"]), {})[str(record["backend"])] = float(record["mean_ns"])
    return rows


def speedup(baseline: float, value: float) -> float:
    return baseline / value if value > 0 else float("nan")


def save_csv(records: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("backend", "category", "problem", "short", "mean_ns", "mean_us"))
        for record in records:
            writer.writerow(
                (
                    record["backend"],
                    record["category"],
                    record["problem"],
                    record["short"],
                    f"{record['mean_ns']:.1f}",
                    f"{record['mean_ns'] / 1e3:.3f}",
                )
            )


def save_iterations_csv(iterations: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("backend", "category", "problem", "iterations", "converged", "final_cost"))
        for item in iterations:
            writer.writerow(
                (
                    item["backend"],
                    item["category"],
                    item["problem"],
                    item["iterations"],
                    "ok" if item["converged"] else "no",
                    "" if item.get("final_cost") is None else f"{item['final_cost']:.6e}",
                )
            )


def plot_group(records: list[dict], title: str, output: Path) -> tuple[str, float]:
    rows = table_rows(records)
    labels = sorted(
        rows,
        key=lambda label: (
            min(int(r["order"]) for r in records if r["short"] == label),
            label,
        ),
    )
    present = [b for b in BACKENDS if any(b in rows[label] for label in labels)]
    peak = max(value for row in rows.values() for value in row.values())
    scale, unit = (1e6, "ms") if peak >= 1e6 else (1e3, "µs")
    figure, axis = plt.subplots(figsize=(max(11, len(present) * 2.2), max(4.5, len(labels) * 0.42)))
    width = 0.8 / max(len(present), 1)
    centers = list(range(len(labels)))
    for index, backend in enumerate(present):
        offsets = [center - 0.4 + width * (index + 0.5) for center in centers]
        values = [rows[label].get(backend, float("nan")) / scale for label in labels]
        axis.barh(offsets, values, height=width * 0.9, label=backend)
    axis.set_xscale("log")
    axis.set_xlabel(f"Mean runtime ({unit}, log scale)")
    axis.set_title(title)
    axis.set_yticks(centers)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.grid(axis="x", which="both", alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return unit, scale


def timing_table(records: list[dict], unit: str, scale: float, with_category: bool) -> str:
    rows = table_rows(records)
    order = {r["short"]: int(r["order"]) for r in records}
    category = {r["short"]: str(r["category"]) for r in records}
    labels = sorted(rows, key=lambda label: (order[label], label))
    if with_category:
        labels.sort(key=lambda label: (0 if category[label] == "Robust pose" else 1, order[label], label))
    headers = (["Problem", "Setup"] if with_category else ["Problem"]) + list(BACKENDS)
    body = []
    for label in labels:
        values = rows[label]
        present = sorted(values.values())
        fastest = present[0] if present else math.nan
        runner = present[1] if len(present) > 1 else math.nan
        baseline = values.get("tinyopt-rs", math.nan)
        cells = [category[label], label] if with_category else [label]
        for backend in BACKENDS:
            value = values.get(backend)
            if value is None:
                cells.append("—")
                continue
            text = f"{value / scale:.2f}"
            if value == fastest:
                text = f"<strong>{text}</strong>"
            elif value == runner:
                text = f"<em>{text}</em>"
            if backend != "tinyopt-rs" and math.isfinite(baseline):
                gain = speedup(baseline, value)
                color = "#16803c" if gain > 1 else "#c62828"
                text += f' (<span style="color:{color}">{gain:.2f}x</span>)'
            cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(
            (f'<th scope="row">{html.escape(c)}</th>' if i < 2 else c)
            for i, c in enumerate(cells)
        ) + "</tr>")
    gains: list[float] = []
    for values in rows.values():
        baseline = values.get("tinyopt-rs")
        for backend in BACKENDS:
            if backend != "tinyopt-rs" and baseline and values.get(backend):
                gains.append(math.log(speedup(baseline, values[backend])))
    mean_gain = math.exp(sum(gains) / len(gains)) if gains else float("nan")
    average = (["—", "Mean speedup"] if with_category else ["Mean speedup"]) + [
        "—" if backend == "tinyopt-rs" else f"{mean_gain:.2f}x" for backend in BACKENDS
    ]
    body.append('<tr class="average-row">' + "".join(f"<th>{c}</th>" if i < (2 if with_category else 1) else f"<td>{c}</td>" for i, c in enumerate(average)) + "</tr>")
    return (
        f'<table><caption>Mean runtime ({unit})</caption><thead><tr>'
        + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def iteration_table(iterations: list[dict]) -> str:
    rows: dict[tuple[str, str], dict[str, dict]] = {}
    for item in iterations:
        rows.setdefault((str(item["category"]), str(item["problem"])), {})[str(item["backend"])] = item
    ordered = sorted(rows, key=lambda key: (GROUP_ORDER.index(key[0]) if key[0] in GROUP_ORDER else 9, key[1]))
    headers = ["Problem", "Setup", *BACKENDS]
    body = []
    for category, problem in ordered:
        cells = [category, problem]
        for backend in BACKENDS:
            item = rows[(category, problem)].get(backend)
            if item is None:
                cells.append("—")
            else:
                cells.append(f"{item['iterations']} {'🎯' if item['converged'] else '❌'}")
        body.append("<tr>" + "".join(
            (f'<th scope="row">{html.escape(c)}</th>' if i < 2 else f"<td>{c}</td>")
            for i, c in enumerate(cells)
        ) + "</tr>")
    counts = []
    for backend in BACKENDS:
        values = [r[backend]["iterations"] for r in rows.values() if backend in r and r[backend]["converged"]]
        counts.append(f"{sum(values) / len(values):.2f}" if values else "—")
    body.append('<tr class="average-row"><th>—</th><th>Mean converged iterations</th>' + "".join(f"<td>{c}</td>" for c in counts) + "</tr>")
    return (
        "<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead><tbody>"
        + "".join(body) + "</tbody></table>"
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    def clean(value: str) -> str:
        return value.replace("|", r"\|").replace("<strong>", "**").replace("</strong>", "**").replace("<em>", "*").replace("</em>", "*")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(clean(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def markdown_timing(records: list[dict], unit: str, scale: float, with_category: bool) -> str:
    rows = table_rows(records)
    order = {r["short"]: int(r["order"]) for r in records}
    category = {r["short"]: str(r["category"]) for r in records}
    labels = sorted(rows, key=lambda label: (order[label], label))
    headers = (["Problem", "Setup"] if with_category else ["Problem"]) + list(BACKENDS)
    body = []
    for label in labels:
        values = rows[label]
        present = sorted(values.values())
        cells = [category[label], label] if with_category else [label]
        for backend in BACKENDS:
            value = values.get(backend)
            if value is None:
                cells.append("—")
                continue
            text = f"{value / scale:.2f}"
            if present and value == present[0]:
                text = f"**{text}**"
            elif len(present) > 1 and value == present[1]:
                text = f"*{text}*"
            baseline = values.get("tinyopt-rs")
            if backend != "tinyopt-rs" and baseline:
                text += f" ({speedup(baseline, value):.2f}x)"
            cells.append(text)
        body.append(cells)
    return f"Mean runtime ({unit})\n\n" + markdown_table(headers, body)


def write_markdown(data: dict, records: list[dict], iterations: list[dict], out: Path) -> Path:
    sections = ["# Tinyopt benchmark report", "", "## Timings", "",
                "Times are means; **bold** is the fastest result and *italics* the runner-up. "
                "Speedups compare each backend with tinyopt-rs.", ""]
    groups = [
        ("Dense static", "Dense static", False),
        ("Dense dynamic", "Dense dynamic", False),
        ("Sparse", "Sparse", False),
        ("Bundle adjustment + robust pose", "Bundle adjustment", True),
    ]
    for index, (title, category, with_category) in enumerate(groups, start=1):
        grouped = [
            r for r in records
            if r["category"] == category
            or (with_category and r["category"] == "Robust pose")
        ]
        if not grouped:
            continue
        unit, scale = plot_group(grouped, title, out / f"benchmark-{index}.png")
        sections += [f"### {title}", "", f"![{title}](benchmark-{index}.png)", "",
                     markdown_timing(grouped, unit, scale, with_category), ""]
    sections += ["## Iterations", "", "🎯 converged, ❌ did not converge.", ""]
    rows: dict[tuple[str, str], dict[str, dict]] = {}
    for item in iterations:
        rows.setdefault((str(item["category"]), str(item["problem"])), {})[str(item["backend"])] = item
    body = []
    for (category, problem) in sorted(rows, key=lambda k: (GROUP_ORDER.index(k[0]) if k[0] in GROUP_ORDER else 9, k[1])):
        cells = [category, problem]
        for backend in BACKENDS:
            item = rows[(category, problem)].get(backend)
            cells.append("—" if item is None else f"{item['iterations']} {'🎯' if item['converged'] else '❌'}")
        body.append(cells)
    sections += [markdown_table(["Problem", "Setup", *BACKENDS], body), ""]
    sections += ["## Notes", ""] + [f"- {note}" for note in data.get("notes", [])] + [""]
    sections += ["## Configuration", ""]
    for key, value in data["meta"].items():
        if key == "backends":
            for name, detail in value.items():
                sections.append(f"- **{name}:** {detail}")
        else:
            sections.append(f"- **{key}:** {value}")
    path = out / "benchmark-report.md"
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return path


def write_html(data: dict, records: list[dict], iterations: list[dict], out: Path) -> Path:
    sections = []
    groups = [
        ("Dense static", "Dense static", False),
        ("Dense dynamic", "Dense dynamic", False),
        ("Sparse", "Sparse", False),
        ("Bundle adjustment + robust pose", "Bundle adjustment", True),
    ]
    for index, (title, category, with_category) in enumerate(groups, start=1):
        grouped = [
            r for r in records
            if r["category"] == category
            or (with_category and r["category"] == "Robust pose")
        ]
        if not grouped:
            continue
        plot = out / f"benchmark-{index}.png"
        unit, scale = plot_group(grouped, title, plot)
        encoded = base64.b64encode(plot.read_bytes()).decode("ascii")
        sections.append(
            f"<details open><summary><h2>{html.escape(title)}</h2></summary>"
            f'<img alt="{html.escape(title)}" src="data:image/png;base64,{encoded}">'
            f'<div class="table-wrap">{timing_table(grouped, unit, scale, with_category)}</div></details>'
        )
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in data.get("notes", []))
    meta = "".join(f"<dt>{html.escape(str(k))}</dt><dd>{html.escape(str(v))}</dd>" for k, v in data["meta"].items() if k != "backends")
    backends = "".join(
        f"<dt>{html.escape(name)}</dt><dd>{html.escape(detail)}</dd>"
        for name, detail in data["meta"].get("backends", {}).items()
    )
    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Tinyopt benchmark report</title><style>"
        "body{font:15px system-ui,sans-serif;color:#20242a;max-width:1440px;margin:32px auto;padding:0 22px;background:#f6f8fa}"
        "h1,h2{color:#172b4d}details{background:white;padding:18px 22px;margin:16px 0;border:1px solid #d8dee4;border-radius:10px;box-shadow:0 2px 8px #172b4d0d}"
        "summary{cursor:pointer;list-style:none}summary::-webkit-details-marker{display:none}"
        "summary:before{content:'▸';display:inline-block;margin-right:8px;transition:transform .15s}details[open]>summary:before{transform:rotate(90deg)}"
        "summary h1,summary h2{display:inline}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}"
        "th,td{padding:9px 12px;border-bottom:1px solid #e6e9ed;text-align:right;white-space:nowrap}th{background:#f0f4f8;color:#334155}"
        "th:first-child,td:first-child{text-align:left}caption{caption-side:top;text-align:left;font-weight:600;padding:8px 0}"
        "img{width:100%;height:auto;border-radius:6px}tr.average-row{border-top:2px solid #9aa7b5}dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 16px}"
        "dt{font-weight:600}dd{margin:0;overflow-wrap:anywhere}"
        "</style></head><body><h1>Tinyopt benchmark report</h1>"
        "<details open><summary><p>Times are means; bold marks the fastest result and italics the runner-up. "
        "Speedups compare each backend with tinyopt-rs.</p></summary></details>"
        + "".join(sections)
        + "<details open><summary><h1>Iterations</h1></summary><div class=\"table-wrap\">"
        + iteration_table(iterations)
        + "</div></details><details open><summary><h1>Notes</h1></summary><ul>"
        + notes
        + "</ul></details><details><summary><h2>Configuration</h2></summary><dl>"
        + meta
        + backends
        + "</dl></details></body></html>"
    )
    path = out / "benchmark-report.html"
    path.write_text(document, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to results.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Report directory")
    parser.add_argument("--format", choices=("html", "md", "both"), default="html")
    parser.add_argument("--open", action="store_true", help="Open the HTML report")
    parser.add_argument("--plot", action="store_true", help="Show the plots interactively")
    args = parser.parse_args()

    out = args.output_dir or args.input.parent / "report"
    out.mkdir(parents=True, exist_ok=True)
    data, records = load(args.input)
    save_csv(records, out / "benchmark-results.csv")
    save_iterations_csv(data.get("iterations", []), out / "benchmark-iterations.csv")
    iterations = data.get("iterations", [])
    if args.format in ("html", "both"):
        report = write_html(data, records, iterations, out)
        print(f"Saved {report}")
    if args.format in ("md", "both"):
        report = write_markdown(data, records, iterations, out)
        print(f"Saved {report}")
    if args.plot:
        plt.show(block=True)
    if args.open and args.format in ("html", "both"):
        webbrowser.open((out / "benchmark-report.html").resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
