#!/usr/bin/env python3
"""
draw_scenario.py - Visualize WTA scenario geometry on a 2D XY map.

Coordinate frame (from scenario generator):
  X = East (km), Y = North (km), Z = Up (km)

Features:
- Vessel positions with defense-radius circles
- Target positions colored by target type
- Target velocity arrows based on VX, VY and Speed
- Optional lines for (weapon, target) pairs that have engagement windows
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


def _target_type_lookup(data):
	infos = data.get("target_infos", [])
	by_code = {}
	for row in infos:
		code = row.get("Code")
		if code:
			by_code[code] = row
	return by_code


def _vessel_lookup(vessels):
	return {v["ID"]: v for v in vessels}


def _window_links_by_target(data):
	"""Return mapping target_id -> list[(weapon_id, vessel_id)] for windowed pairs."""
	weapons = data["assignment_request"].get("weapons", [])
	wid_to_vid = {w["ID"]: w["WTAVesselID"] for w in weapons}

	links = defaultdict(list)
	for key in data.get("engagement_windows", {}).keys():
		wid_str, tid_str = key.split("_")
		wid = int(wid_str)
		tid = int(tid_str)
		vid = wid_to_vid.get(wid)
		if vid is not None:
			links[tid].append((wid, vid))
	return links


def plot_scenario(
	scenario_path: str,
	out_path: str | None = None,
	show_windows: bool = True,
	arrow_seconds: float = 4.0,
	pad_km: float = 2.0,
	trajectory_seconds: float = 20.0,
	marker_step: float = 5.0,
	show_time_markers: bool = True,
):
	with open(scenario_path) as f:
		data = json.load(f)

	req = data["assignment_request"]
	vessels = req.get("vessels", [])
	targets = req.get("targets", [])

	if not vessels:
		raise ValueError("No vessels found in scenario")
	if not targets:
		raise ValueError("No targets found in scenario")

	vessels_by_id = _vessel_lookup(vessels)
	type_info = _target_type_lookup(data)
	links_by_target = _window_links_by_target(data) if show_windows else {}

	traj_points_x = []
	traj_points_y = []

	fig, ax = plt.subplots(figsize=(10, 9))

	# Vessel markers and defense circles
	for vessel in vessels:
		x, y = vessel["X"], vessel["Y"]
		vid = vessel["ID"]
		dr = vessel.get("DefenseRadius", 0.0)
		spd_km_s = float(vessel.get("Speed", 0.0)) * 1e-3
		vx = float(vessel.get("HeadingX", 0.0)) * spd_km_s
		vy = float(vessel.get("HeadingY", 0.0)) * spd_km_s

		ax.scatter(x, y, marker="^", s=220, c="#0b7285", edgecolors="black", linewidths=0.8, zorder=5)
		ax.text(x + 0.12, y + 0.12, f"V{vid}", fontsize=10, weight="bold", zorder=6)

		if dr and dr > 0:
			circ = plt.Circle((x, y), dr, color="#0b7285", alpha=0.10, lw=1.2, fill=True)
			ax.add_patch(circ)
			ring = plt.Circle((x, y), dr, color="#0b7285", alpha=0.45, lw=1.2, fill=False, linestyle="--")
			ax.add_patch(ring)

		# Predicted vessel trajectory over the selected horizon.
		x_end = x + vx * trajectory_seconds
		y_end = y + vy * trajectory_seconds
		ax.plot([x, x_end], [y, y_end], color="#0b7285", linestyle=":", linewidth=1.4, alpha=0.8, zorder=4)
		ax.scatter(x_end, y_end, marker="x", s=60, c="#0b7285", linewidths=1.5, zorder=6)

		traj_points_x.extend([x_end])
		traj_points_y.extend([y_end])

		if show_time_markers and marker_step > 0:
			t = marker_step
			while t < trajectory_seconds:
				tx = x + vx * t
				ty = y + vy * t
				ax.scatter(tx, ty, marker="+", s=30, c="#0b7285", alpha=0.55, zorder=5)
				t += marker_step

	# Consistent color by target type code
	t_codes = sorted({t["WTATargetInfoCode"] for t in targets})
	palette = plt.get_cmap("tab10")
	tcode_color = {code: palette(i % 10) for i, code in enumerate(t_codes)}

	# Target markers, labels, and velocity arrows
	for tgt in targets:
		tid = tgt["ID"]
		x, y = tgt["X"], tgt["Y"]
		tcode = tgt["WTATargetInfoCode"]
		color = tcode_color[tcode]

		ax.scatter(x, y, marker="o", s=110, c=[color], edgecolors="black", linewidths=0.6, zorder=7)

		threat = tgt.get("ThreatScore", None)
		threat_txt = f" {threat:.2f}" if isinstance(threat, (int, float)) else ""
		ax.text(x + 0.10, y + 0.10, f"T{tid}{threat_txt}", fontsize=9, zorder=8)

		# VX/VY are unit direction components; Speed is in m/s.
		spd_km_s = float(tgt.get("Speed", 0.0)) * 1e-3
		dx = float(tgt.get("VX", 0.0)) * spd_km_s * arrow_seconds
		dy = float(tgt.get("VY", 0.0)) * spd_km_s * arrow_seconds
		ax.arrow(
			x,
			y,
			dx,
			dy,
			length_includes_head=True,
			head_width=0.08,
			head_length=0.12,
			linewidth=1.2,
			color=color,
			alpha=0.85,
			zorder=6,
		)

		# Predicted target trajectory over the selected horizon.
		tx_end = x + float(tgt.get("VX", 0.0)) * spd_km_s * trajectory_seconds
		ty_end = y + float(tgt.get("VY", 0.0)) * spd_km_s * trajectory_seconds
		ax.plot([x, tx_end], [y, ty_end], color=color, linestyle="--", linewidth=1.2, alpha=0.6, zorder=4)
		ax.scatter(tx_end, ty_end, marker="x", s=45, c=[color], linewidths=1.2, zorder=6)

		traj_points_x.extend([tx_end])
		traj_points_y.extend([ty_end])

		if show_time_markers and marker_step > 0:
			t = marker_step
			while t < trajectory_seconds:
				tx = x + float(tgt.get("VX", 0.0)) * spd_km_s * t
				ty = y + float(tgt.get("VY", 0.0)) * spd_km_s * t
				ax.scatter(tx, ty, marker=".", s=16, c=[color], alpha=0.45, zorder=5)
				t += marker_step

		if show_windows and tid in links_by_target:
			for _wid, vid in links_by_target[tid]:
				v = vessels_by_id.get(vid)
				if not v:
					continue
				ax.plot([v["X"], x], [v["Y"], y], color=color, alpha=0.20, linewidth=1.0, zorder=3)

	# Axes, grid, and title
	ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
	ax.axvline(0, color="gray", linewidth=0.8, alpha=0.6)
	ax.set_xlabel("X (km, East)")
	ax.set_ylabel("Y (km, North)")
	ax.set_title("WTA Scenario Map")
	ax.grid(True, linestyle="--", alpha=0.35)

	# Zoom to the tactical objects (vessels + targets), not to defense circles.
	all_x = [v["X"] for v in vessels] + [t["X"] for t in targets] + traj_points_x
	all_y = [v["Y"] for v in vessels] + [t["Y"] for t in targets] + traj_points_y
	x_min, x_max = min(all_x), max(all_x)
	y_min, y_max = min(all_y), max(all_y)

	span_x = x_max - x_min
	span_y = y_max - y_min
	half_span = max(span_x, span_y, 1.0) / 2.0 + pad_km
	cx = (x_min + x_max) / 2.0
	cy = (y_min + y_max) / 2.0

	ax.set_xlim(cx - half_span, cx + half_span)
	ax.set_ylim(cy - half_span, cy + half_span)
	ax.set_aspect("equal", adjustable="box")

	# Legends
	type_counts = Counter(t["WTATargetInfoCode"] for t in targets)
	type_labels = []
	for code in t_codes:
		info = type_info.get(code, {})
		desc = info.get("Description", code)
		type_labels.append(f"{code} ({type_counts[code]}): {desc}")

	legend_lines = [
		f"Vessels: {len(vessels)}",
		f"Targets: {len(targets)}",
		f"Window links: {sum(len(v) for v in links_by_target.values()) if show_windows else 0}",
		f"Trajectory horizon: {trajectory_seconds:g}s",
	]
	summary = "\n".join(legend_lines + ["", "Target types:"] + type_labels)
	ax.text(
		1.01,
		1.0,
		summary,
		transform=ax.transAxes,
		va="top",
		ha="left",
		fontsize=9,
		bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"),
	)

	plt.tight_layout()

	if out_path:
		plt.savefig(out_path, dpi=160, bbox_inches="tight")
		print(f"Saved map to {out_path}")
	else:
		plt.show()


def main():
	parser = argparse.ArgumentParser(description="Draw a WTA scenario on a 2D map")
	parser.add_argument(
		"scenario",
		nargs="?",
		default="/workspaces/WTA/data/scenario_001.json",
		help="Path to scenario JSON",
	)
	parser.add_argument(
		"--out",
		default=None,
		help="Save figure to this path instead of opening an interactive window",
	)
	parser.add_argument(
		"--no-window-links",
		action="store_true",
		help="Do not draw vessel-target lines for engagement-window pairs",
	)
	parser.add_argument(
		"--arrow-seconds",
		type=float,
		default=4.0,
		help="Arrow length in seconds of target travel (default: 4.0)",
	)
	parser.add_argument(
		"--pad-km",
		type=float,
		default=2.0,
		help="Extra map padding in km around vessels/targets (default: 2.0)",
	)
	parser.add_argument(
		"--trajectory-seconds",
		type=float,
		default=20.0,
		help="Prediction horizon in seconds for trajectory overlays (default: 20)",
	)
	parser.add_argument(
		"--marker-step",
		type=float,
		default=5.0,
		help="Time spacing in seconds for trajectory markers (default: 5)",
	)
	parser.add_argument(
		"--no-time-markers",
		action="store_true",
		help="Disable intermediate time markers along trajectories",
	)
	args = parser.parse_args()

	if args.arrow_seconds <= 0:
		raise ValueError("--arrow-seconds must be positive")
	if args.pad_km < 0:
		raise ValueError("--pad-km must be non-negative")
	if args.trajectory_seconds <= 0:
		raise ValueError("--trajectory-seconds must be positive")
	if args.marker_step <= 0:
		raise ValueError("--marker-step must be positive")

	plot_scenario(
		scenario_path=args.scenario,
		out_path=args.out,
		show_windows=not args.no_window_links,
		arrow_seconds=args.arrow_seconds,
		pad_km=args.pad_km,
		trajectory_seconds=args.trajectory_seconds,
		marker_step=args.marker_step,
		show_time_markers=not args.no_time_markers,
	)


if __name__ == "__main__":
	main()
