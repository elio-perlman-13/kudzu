#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from statistics import mean


def load_metrics(path: Path):
	rows = {}
	with path.open(newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		required = {"scenario", "best", "mean"}
		missing = required - set(reader.fieldnames or [])
		if missing:
			raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
		for row in reader:
			scenario = row["scenario"].strip()
			rows[scenario] = {
				"best": float(row["best"]),
				"mean": float(row["mean"]),
			}
	return rows


def pct_gap(other: float, baseline: float):
	if baseline == 0.0:
		return float("nan")
	return (other - baseline) / baseline * 100.0


def main():
	parser = argparse.ArgumentParser(
		description=(
			"Compare GRASP metric files against a chronology baseline. "
			"Outputs per-scenario best/mean gap(%) and overall averages."
		)
	)
	parser.add_argument(
		"--baseline",
		default="grasp_metrics_all_scenarios (chronology).csv",
		help="Baseline CSV path (default: chronology file)",
	)
	parser.add_argument(
		"--effectiveness",
		default="grasp_metrics_all_scenarios (effectiveness).csv",
		help="Effectiveness CSV path",
	)
	parser.add_argument(
		"--firetime",
		default="grasp_metrics_all_scenarios (firetime).csv",
		help="Firetime CSV path",
	)
	parser.add_argument(
		"--scarcity",
		default="grasp_metrics_all_scenarios (scarcity).csv",
		help="Scarcity CSV path",
	)
	parser.add_argument(
		"--out",
		default="grasp_vs_chronology_gaps.csv",
		help="Output CSV path for per-scenario + average rows",
	)
	args = parser.parse_args()

	baseline = load_metrics(Path(args.baseline))
	variants = {
		"effectiveness": load_metrics(Path(args.effectiveness)),
		"firetime": load_metrics(Path(args.firetime)),
		"scarcity": load_metrics(Path(args.scarcity)),
	}

	shared = set(baseline.keys())
	for _, data in variants.items():
		shared &= set(data.keys())
	scenarios = sorted(shared)
	if not scenarios:
		raise ValueError("No shared scenarios across baseline and variant files.")

	fields = ["scenario"]
	for name in ("effectiveness", "firetime", "scarcity"):
		fields.extend([f"{name}_best_gap_pct", f"{name}_mean_gap_pct"])

	rows = []
	for scenario in scenarios:
		row = {"scenario": scenario}
		b_best = baseline[scenario]["best"]
		b_mean = baseline[scenario]["mean"]
		for name in ("effectiveness", "firetime", "scarcity"):
			row[f"{name}_best_gap_pct"] = pct_gap(variants[name][scenario]["best"], b_best)
			row[f"{name}_mean_gap_pct"] = pct_gap(variants[name][scenario]["mean"], b_mean)
		rows.append(row)

	avg_row = {"scenario": "AVERAGE_ALL_SCENARIOS"}
	for name in ("effectiveness", "firetime", "scarcity"):
		avg_row[f"{name}_best_gap_pct"] = mean(r[f"{name}_best_gap_pct"] for r in rows)
		avg_row[f"{name}_mean_gap_pct"] = mean(r[f"{name}_mean_gap_pct"] for r in rows)

	out_path = Path(args.out)
	with out_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fields)
		writer.writeheader()
		writer.writerows(rows)
		writer.writerow(avg_row)

	print(f"Compared {len(scenarios)} shared scenarios.")
	for name in ("effectiveness", "firetime", "scarcity"):
		print(
			f"{name:13s} avg best gap (%): {avg_row[f'{name}_best_gap_pct']:.4f} | "
			f"avg mean gap (%): {avg_row[f'{name}_mean_gap_pct']:.4f}"
		)
	print(f"Per-scenario results written to {out_path}")


if __name__ == "__main__":
	main()
