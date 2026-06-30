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
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


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


def _weapon_info_lookup(data):
	return {
		row["Code"]: row
		for row in data.get("weapon_infos", [])
		if row.get("Code")
	}


def _target_color_map(targets):
	"""Use exactly the same target-ID colour mapping as plot.py."""
	all_tids = sorted(t["ID"] for t in targets)
	cmap = (
		matplotlib.colormaps["tab20"]
		if len(all_tids) <= 20
		else matplotlib.colormaps["hsv"]
	)
	return {
		tid: cmap(i / max(len(all_tids) - 1, 1))
		for i, tid in enumerate(all_tids)
	}


def _heading_bearing_deg(vessel):
	"""Heading bearing in degrees clockwise from North."""
	hx = float(vessel.get("HeadingX", 0.0))
	hy = float(vessel.get("HeadingY", 1.0))
	if math.hypot(hx, hy) <= 1e-12:
		return 0.0
	return math.degrees(math.atan2(hx, hy)) % 360.0


def _draw_weapon_azimuth_sector(
	ax,
	vessel,
	weapon,
	weapon_info,
	color,
):
	"""Draw one weapon's relative azimuth sector around its vessel."""
	x = float(vessel["X"])
	y = float(vessel["Y"])

	heading_bearing = _heading_bearing_deg(vessel)
	rel_from = float(weapon_info.get("AzimuthFromDeg", 0.0)) % 360.0
	rel_to = float(weapon_info.get("AzimuthToDeg", 0.0)) % 360.0

	# The JSON convention uses a clockwise sweep from AzimuthFromDeg
	# to AzimuthToDeg. For example, 320 -> 180 is a 220-degree sector.
	sweep = (rel_to - rel_from) % 360.0
	if sweep <= 1e-9:
		sweep = 360.0

	abs_from = heading_bearing + rel_from
	bearings = np.linspace(abs_from, abs_from + sweep, 181)
	radians = np.deg2rad(bearings)

	# Weapon ranges are stored in metres; the scenario map uses kilometres.
	max_range_km = float(weapon_info.get("MaxRange", 0.0)) * 1e-3
	if max_range_km <= 0.0:
		max_range_km = float(vessel.get("DefenseRadius", 0.0))
	if max_range_km <= 0.0:
		return

	arc_x = x + max_range_km * np.sin(radians)
	arc_y = y + max_range_km * np.cos(radians)

	sector_points = [(x, y), *zip(arc_x, arc_y)]
	sector = mpatches.Polygon(
		sector_points,
		closed=True,
		facecolor=(*color[:3], 0.055),
		edgecolor="none",
		zorder=1,
	)
	ax.add_patch(sector)

	# Arc and two boundary rays.
	ax.plot(
		arc_x,
		arc_y,
		color=color,
		linewidth=1.5,
		linestyle="-.",
		alpha=0.85,
		zorder=2,
	)
	for bearing in (abs_from, abs_from + sweep):
		rad = math.radians(bearing)
		bx = x + max_range_km * math.sin(rad)
		by = y + max_range_km * math.cos(rad)
		ax.plot(
			[x, bx],
			[y, by],
			color=color,
			linewidth=1.0,
			linestyle=":",
			alpha=0.8,
			zorder=2,
		)

	# Label near the middle of the azimuth sector.
	mid_bearing = math.radians(abs_from + sweep / 2.0)
	label_radius = 0.72 * max_range_km
	label_x = x + label_radius * math.sin(mid_bearing)
	label_y = y + label_radius * math.cos(mid_bearing)
	ax.text(
		label_x,
		label_y,
		f"W{weapon['ID']}\n{rel_from:g}°→{rel_to:g}°",
		ha="center",
		va="center",
		fontsize=8,
		color=color,
		weight="bold",
		bbox=dict(
			facecolor="white",
			alpha=0.72,
			edgecolor=color,
			linewidth=0.6,
			boxstyle="round,pad=0.2",
		),
		zorder=9,
	)


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
	weapons = req.get("weapons", [])
	targets = req.get("targets", [])

	if not vessels:
		raise ValueError("No vessels found in scenario")
	if not targets:
		raise ValueError("No targets found in scenario")

	vessels_by_id = _vessel_lookup(vessels)
	type_info = _target_type_lookup(data)
	weapon_info = _weapon_info_lookup(data)
	links_by_target = _window_links_by_target(data) if show_windows else {}

	traj_points_x = []
	traj_points_y = []

	fig, ax = plt.subplots(figsize=(12, 9))

	# Vessel markers and defense circles
	for vessel in vessels:
		x, y = vessel["X"], vessel["Y"]
		vid = vessel["ID"]
		dr = vessel.get("DefenseRadius", 0.0)
		spd_km_s = float(vessel.get("Speed", 0.0)) * 1e-3
		vx = float(vessel.get("HeadingX", 0.0)) * spd_km_s
		vy = float(vessel.get("HeadingY", 0.0)) * spd_km_s

		ax.scatter(x, y, marker="o", s=220, c="#0b7285", edgecolors="black", linewidths=0.8, zorder=5)
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

	# Draw each weapon's azimuth sector using the vessel heading as the
	# zero-bearing reference. Weapon colours are distinct from target colours.
	weapon_palette = matplotlib.colormaps["Dark2"]
	weapon_colors = {
		weapon["ID"]: weapon_palette(i / max(len(weapons) - 1, 1))
		for i, weapon in enumerate(sorted(weapons, key=lambda w: w["ID"]))
	}
	for weapon in weapons:
		vessel = vessels_by_id.get(weapon["WTAVesselID"])
		info = weapon_info.get(weapon["WTAWeaponInfoCode"])
		if vessel is None or info is None:
			continue
		_draw_weapon_azimuth_sector(
			ax,
			vessel,
			weapon,
			info,
			weapon_colors[weapon["ID"]],
		)

	# Use the same per-target ID colour mapping as plot.py.
	t_codes = sorted({t["WTATargetInfoCode"] for t in targets})
	tid_color = _target_color_map(targets)

	# Target markers, labels, and velocity arrows
	for tgt in targets:
		tid = tgt["ID"]
		x, y = tgt["X"], tgt["Y"]
		tcode = tgt["WTATargetInfoCode"]
		color = tid_color[tid]

		ax.scatter(x, y, marker="^", s=120, c=[color], edgecolors="black", linewidths=0.6, zorder=7)

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

	# Legend: keep target colours synchronized with plot.py and include
	# target speed, weapon count, and weapon azimuth information.
	legend_handles = [
		Line2D(
			[0],
			[0],
			marker="o",
			color="none",
			markerfacecolor="#0b7285",
			markeredgecolor="black",
			markersize=10,
			label=f"Vessels: {len(vessels)}",
		)
	]

	for tgt in sorted(targets, key=lambda row: row["ID"]):
		tid = tgt["ID"]
		code = tgt["WTATargetInfoCode"]
		desc = type_info.get(code, {}).get("Description", code)
		if "YJ83" in code.upper() or "YJ-83" in desc.upper():
			short_desc = "YJ-83"
		elif "UAV" in code.upper() or "UAV" in desc.upper():
			short_desc = "UAV"
		else:
			short_desc = desc
		speed = float(tgt.get("Speed", 0.0))
		legend_handles.append(
			Line2D(
				[0],
				[0],
				marker="^",
				color="none",
				markerfacecolor=tid_color[tid],
				markeredgecolor="black",
				markersize=8,
				label=f"T{tid} {short_desc}: {speed:.1f} m/s",
			)
		)

	for weapon in sorted(weapons, key=lambda row: row["ID"]):
		info = weapon_info.get(weapon["WTAWeaponInfoCode"], {})
		az_from = float(info.get("AzimuthFromDeg", 0.0)) % 360.0
		az_to = float(info.get("AzimuthToDeg", 0.0)) % 360.0
		legend_handles.append(
			Line2D(
				[0],
				[0],
				color=weapon_colors.get(weapon["ID"], "black"),
				linestyle="-.",
				linewidth=1.8,
				label=f"W{weapon['ID']} az. {az_from:g}°→{az_to:g}°",
			)
		)

	legend = ax.legend(
		handles=legend_handles,
		loc="upper right",
		borderaxespad=0.6,
		fontsize=8.5,
		title=f"Objects ({len(weapons)} weapons)",
		title_fontsize=9,
		framealpha=0.9,
	)
	ax.add_artist(legend)

	window_count = (
		sum(len(rows) for rows in links_by_target.values())
		if show_windows
		else 0
	)
	summary = (
		f"Targets: {len(targets)}\n"
		f"Window links: {window_count}\n"
		f"Trajectory horizon: {trajectory_seconds:g}s"
	)
	ax.text(
		0.99,
		0.34,
		summary,
		transform=ax.transAxes,
		va="top",
		ha="right",
		fontsize=8.5,
		bbox=dict(
			facecolor="white",
			alpha=0.85,
			edgecolor="#cccccc",
		),
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
