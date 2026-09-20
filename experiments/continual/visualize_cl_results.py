#!/usr/bin/env python3
"""Build a dependency-free SVG dashboard from continual-learning metrics JSON."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path


CORE_METRIC = re.compile(
    r"^val-core/(?P<dataset>[^/]+)/acc/(?P<stat>mean@\d+|maj@\d+/mean|best@\d+/mean)$"
)
STAGE_NAME = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)-(?P<task>[^/]+)$")
COLORS = {
    "math": "#e45756",
    "math500": "#f2a541",
    "gpqa": "#4c78a8",
    "sciknoweval": "#72b7b2",
    "tooluse": "#785ef0",
}
FALLBACK_COLORS = ["#54a24b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac"]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.glob("*/*.metrics.json")):
        match = STAGE_NAME.match(path.parent.name)
        if not match:
            continue
        payload = json.loads(path.read_text())
        scores: dict[str, dict[str, float]] = {}
        for key, value in payload.get("metrics", {}).items():
            metric_match = CORE_METRIC.match(key)
            if metric_match:
                scores.setdefault(metric_match["dataset"], {})[metric_match["stat"]] = float(value)
        if scores:
            runs.append(
                {
                    "stage": int(match["number"]),
                    "task": match["task"],
                    "name": path.parent.name,
                    "step": int(payload.get("step", 0)),
                    "num_samples": int(payload.get("num_samples", 0)),
                    "scores": scores,
                    "source": str(path),
                }
            )
    if not runs:
        raise ValueError(f"No val-core metrics found below {root}")
    return sorted(runs, key=lambda row: row["stage"])


def stat_key(scores: dict[str, float], prefix: str) -> str | None:
    keys = [key for key in scores if key.startswith(prefix)]
    if not keys:
        return None
    return max(keys, key=lambda key: int(re.search(r"@(\d+)", key).group(1)))


def primary(scores: dict[str, float]) -> float:
    key = stat_key(scores, "mean@")
    if key is None:
        raise ValueError("Dataset is missing mean@N")
    return scores[key]


def color_for(dataset: str, index: int) -> str:
    return COLORS.get(dataset, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def text(x: float, y: float, value: object, size: int = 14, fill: str = "#273142", **attrs: object) -> str:
    def svg_attr(key: str) -> str:
        key = key[:-1] if key.endswith("_") else key
        return key.replace("_", "-")

    extra = " ".join(f'{svg_attr(key)}="{esc(val)}"' for key, val in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" {extra}>{esc(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#d9dee8", width: float = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"/>'


def make_svg(runs: list[dict], output: Path) -> None:
    datasets = list(dict.fromkeys(dataset for run in runs for dataset in run["scores"]))
    first_seen = {dataset: next(i for i, run in enumerate(runs) if dataset in run["scores"]) for dataset in datasets}
    final = runs[-1]
    width, height = 1280, 920
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f8fc"/>',
        '<style>text{font-family:Inter,Arial,"Noto Sans",sans-serif}.bold{font-weight:700}.mono{font-family:monospace}</style>',
        text(48, 52, "SDPO-SVC Continual Learning Dashboard", 27, "#172033", class_="bold"),
        text(48, 78, f'{len(runs)} stages · final checkpoint: {final["name"]} · evaluation step {final["step"]}', 14, "#667085"),
    ]

    # Panel 1: mean@N trajectories.
    px, py, pw, ph = 48, 110, 750, 365
    parts.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="14" fill="white"/>')
    parts.append(text(px + 24, py + 34, "Single-sample accuracy (mean@N)", 18, "#172033", class_="bold"))
    parts.append(text(px + 24, py + 56, "Higher is better; missing points mean the benchmark was not evaluated yet.", 12, "#667085"))
    gx0, gy0, gw, gh = px + 62, py + 78, pw - 92, ph - 128
    for tick in range(0, 81, 20):
        y = gy0 + gh * (1 - tick / 80)
        parts.extend([line(gx0, y, gx0 + gw, y), text(gx0 - 10, y + 4, f"{tick}%", 11, "#667085", text_anchor="end")])
    for i, run in enumerate(runs):
        x = gx0 + (gw * i / max(1, len(runs) - 1))
        parts.append(line(x, gy0, x, gy0 + gh, "#eef1f6"))
        parts.append(text(x, gy0 + gh + 27, f'{run["stage"]}. {run["task"]}', 12, "#475467", text_anchor="middle"))
    for d_idx, dataset in enumerate(datasets):
        points = []
        color = color_for(dataset, d_idx)
        for i, run in enumerate(runs):
            if dataset not in run["scores"]:
                continue
            value = primary(run["scores"][dataset])
            x = gx0 + (gw * i / max(1, len(runs) - 1))
            y = gy0 + gh * (1 - value / 0.8)
            points.append((x, y, value))
        if len(points) > 1:
            coords = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y, value in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="white" stroke-width="2"/>')
            parts.append(text(x, y - 10, f"{value:.1%}", 10, color, text_anchor="middle", class_="bold"))
    legend_x, legend_y = px + 28, py + ph - 19
    for d_idx, dataset in enumerate(datasets):
        x = legend_x + d_idx * 130
        parts.append(f'<circle cx="{x}" cy="{legend_y - 4}" r="5" fill="{color_for(dataset, d_idx)}"/>')
        parts.append(text(x + 10, legend_y, dataset, 11, "#475467"))

    # Panel 2: final decoding strategies.
    px, py, pw, ph = 822, 110, 410, 365
    parts.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="14" fill="white"/>')
    parts.append(text(px + 24, py + 34, "Final checkpoint: decoding headroom", 18, "#172033", class_="bold"))
    parts.append(text(px + 24, py + 56, "mean@N vs majority vote vs best-of-N", 12, "#667085"))
    bx0, by0, bw = px + 105, py + 83, pw - 135
    bar_h, gap = 8, 48
    for tick in range(0, 81, 20):
        x = bx0 + bw * tick / 80
        parts.append(line(x, by0 - 8, x, by0 + gap * len(datasets) - 12, "#eef1f6"))
        parts.append(text(x, by0 + gap * len(datasets) + 5, f"{tick}%", 10, "#667085", text_anchor="middle"))
    stat_specs = [("mean@", "#98a2b3"), ("maj@", "#4c78a8"), ("best@", "#72b7b2")]
    for d_idx, dataset in enumerate(datasets):
        y = by0 + d_idx * gap
        parts.append(text(bx0 - 10, y + 10, dataset, 11, "#344054", text_anchor="end"))
        scores = final["scores"].get(dataset, {})
        for s_idx, (prefix, color) in enumerate(stat_specs):
            key = stat_key(scores, prefix)
            if key is None:
                continue
            value = scores[key]
            bar_y = y + s_idx * (bar_h + 2)
            parts.append(f'<rect x="{bx0}" y="{bar_y}" width="{bw * value / 0.8:.1f}" height="{bar_h}" rx="3" fill="{color}"/>')
    for s_idx, (prefix, color) in enumerate(stat_specs):
        lx = px + 112 + s_idx * 86
        parts.append(f'<rect x="{lx}" y="{py + ph - 24}" width="12" height="8" rx="2" fill="{color}"/>')
        parts.append(text(lx + 17, py + ph - 16, {"mean@": "mean", "maj@": "majority", "best@": "best"}[prefix], 10, "#667085"))

    # Panel 3: cumulative forgetting from first observed checkpoint.
    px, py, pw, ph = 48, 500, 750, 370
    parts.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="14" fill="white"/>')
    parts.append(text(px + 24, py + 34, "Cumulative forgetting by final checkpoint", 18, "#172033", class_="bold"))
    parts.append(text(px + 24, py + 56, "First observed mean@N minus final mean@N (percentage points)", 12, "#667085"))
    forget_rows = []
    for d_idx, dataset in enumerate(datasets):
        first = primary(runs[first_seen[dataset]]["scores"][dataset])
        last = primary(final["scores"][dataset])
        forget_rows.append((dataset, (first - last) * 100, first, last, d_idx))
    max_drop = max(30.0, max(drop for _, drop, _, _, _ in forget_rows) * 1.12)
    fx0, fy0, fw = px + 135, py + 84, pw - 180
    for tick in range(0, int(max_drop) + 1, 10):
        x = fx0 + fw * tick / max_drop
        parts.append(line(x, fy0 - 9, x, fy0 + 45 * len(forget_rows), "#eef1f6"))
        parts.append(text(x, fy0 + 45 * len(forget_rows) + 20, f"{tick} pp", 10, "#667085", text_anchor="middle"))
    for row_idx, (dataset, drop, first, last, d_idx) in enumerate(forget_rows):
        y = fy0 + row_idx * 45
        parts.append(text(fx0 - 12, y + 15, dataset, 12, "#344054", text_anchor="end"))
        shown_drop = max(0, drop)
        parts.append(f'<rect x="{fx0}" y="{y}" width="{fw * shown_drop / max_drop:.1f}" height="21" rx="5" fill="{color_for(dataset, d_idx)}" opacity="0.88"/>')
        label_x = fx0 + fw * shown_drop / max_drop + 8
        parts.append(text(label_x, y + 15, f"{drop:+.1f} pp", 11, "#344054", class_="bold"))
        parts.append(text(fx0, y + 34, f"{first:.1%} → {last:.1%}", 9, "#98a2b3"))

    # Panel 4: summary cards and interpretation guardrails.
    px, py, pw, ph = 822, 500, 410, 370
    parts.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="14" fill="white"/>')
    parts.append(text(px + 24, py + 34, "At a glance", 18, "#172033", class_="bold"))
    prior = [row for row in forget_rows if first_seen[row[0]] < len(runs) - 1]
    avg_drop = sum(row[1] for row in prior) / len(prior) if prior else 0.0
    final_avg = sum(primary(final["scores"][dataset]) for dataset in datasets) / len(datasets)
    tool_dataset = next((dataset for dataset in datasets if first_seen[dataset] == len(runs) - 1), datasets[-1])
    cards = [
        ("Final macro average", f"{final_avg:.1%}"),
        ("Avg. prior-benchmark forgetting", f"{avg_drop:.1f} pp"),
        (f"New task ({tool_dataset})", f'{primary(final["scores"][tool_dataset]):.1%}'),
    ]
    for i, (label, value) in enumerate(cards):
        cy = py + 58 + i * 73
        parts.append(f'<rect x="{px + 22}" y="{cy}" width="{pw - 44}" height="60" rx="9" fill="#f8fafc"/>')
        parts.append(text(px + 38, cy + 22, label, 11, "#667085"))
        parts.append(text(px + 38, cy + 48, value, 22, "#172033", class_="bold"))
    note_y = py + 293
    parts.append(text(px + 24, note_y, "Metric guide", 12, "#344054", class_="bold"))
    parts.append(text(px + 24, note_y + 22, "mean@N: expected accuracy of one sampled answer", 10, "#667085"))
    parts.append(text(px + 24, note_y + 39, "majority: answer selected by N-sample voting", 10, "#667085"))
    parts.append(text(px + 24, note_y + 56, "best: oracle selection from N sampled answers", 10, "#667085"))

    parts.append(text(48, 904, "Generated from val-core/*/acc metrics; macro averages weight benchmarks equally.", 10, "#98a2b3"))
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n")


def write_tables(runs: list[dict], output_dir: Path) -> None:
    datasets = list(dict.fromkeys(dataset for run in runs for dataset in run["scores"]))
    with (output_dir / "scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "task", "dataset", "mean", "majority", "best", "rollouts", "source"])
        for run in runs:
            for dataset, scores in run["scores"].items():
                mean_key = stat_key(scores, "mean@")
                maj_key = stat_key(scores, "maj@")
                best_key = stat_key(scores, "best@")
                writer.writerow(
                    [
                        run["stage"], run["task"], dataset, scores.get(mean_key, ""), scores.get(maj_key, ""),
                        scores.get(best_key, ""), int(re.search(r"@(\d+)", mean_key).group(1)), run["source"],
                    ]
                )
    final = runs[-1]
    with (output_dir / "forgetting.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "first_stage", "first_mean", "final_mean", "forgetting_pp", "retention_ratio"])
        for dataset in datasets:
            first_run = next(run for run in runs if dataset in run["scores"])
            first, last = primary(first_run["scores"][dataset]), primary(final["scores"][dataset])
            writer.writerow([dataset, first_run["stage"], first, last, (first - last) * 100, last / first if first else ""])


def write_report(runs: list[dict], output: Path) -> None:
    datasets = list(dict.fromkeys(dataset for run in runs for dataset in run["scores"]))
    final = runs[-1]
    rows = []
    for dataset in datasets:
        first_run = next(run for run in runs if dataset in run["scores"])
        first, last = primary(first_run["scores"][dataset]), primary(final["scores"][dataset])
        rows.append((dataset, first_run["stage"], first, last, (first - last) * 100))
    report = [
        "# Continual-learning evaluation summary",
        "",
        "Primary metric: `val-core/<dataset>/acc/mean@N`, the average score across N sampled responses per prompt.",
        "",
        "| Benchmark | First stage | First mean | Final mean | Forgetting |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, stage, first, last, drop in rows:
        report.append(f"| {dataset} | {stage} | {first:.2%} | {last:.2%} | {drop:+.2f} pp |")
    report.extend(["", "See `dashboard.svg` for trajectories and final-checkpoint decoding headroom.", ""])
    output.write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.evaluation_root.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.evaluation_root)
    make_svg(runs, output_dir / "dashboard.svg")
    write_tables(runs, output_dir)
    write_report(runs, output_dir / "report.md")
    for name in ("dashboard.svg", "scores.csv", "forgetting.csv", "report.md"):
        print(output_dir / name)


if __name__ == "__main__":
    main()
