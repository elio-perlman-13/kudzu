#!/usr/bin/env python3
"""
draw_scenario.py — Draw a WTA scenario map.

This version supports both:
1. Legacy scenarios with one straight-line target trajectory derived from
   X, Y, VX, VY, Speed.
2. Real scenarios containing `target_trajectories`, where each target may
   follow a piecewise waypoint route during the scenario horizon.

For real scenarios, the figure contains:
- an operational overview showing the enemy ship and all incoming targets;
- a close-in defense view around the own ship;
- actual piecewise target trajectories and waypoint-turn times;
- target colors synchronized with plot.py;
- weapon azimuth/range coverage;
- target speeds, arrival times, and weapon information in the legend.

Coordinate frame:
  X = East (km), Y = North (km), Z = Up (km)
  Bearings are clockwise from North.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


EPS = 1e-9


# ---------------------------------------------------------------------------
# Basic lookup and formatting helpers
# ---------------------------------------------------------------------------

def _target_type_lookup(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["Code"]: row
        for row in data.get("target_infos", [])
        if row.get("Code")
    }


def _weapon_info_lookup(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["Code"]: row
        for row in data.get("weapon_infos", [])
        if row.get("Code")
    }


def _target_color_map(targets: Iterable[dict[str, Any]]) -> dict[int, Any]:
    """Use exactly the same target-ID color mapping as plot.py."""
    target_ids = sorted(int(target["ID"]) for target in targets)
    cmap = (
        matplotlib.colormaps["tab20"]
        if len(target_ids) <= 20
        else matplotlib.colormaps["hsv"]
    )
    return {
        target_id: cmap(index / max(len(target_ids) - 1, 1))
        for index, target_id in enumerate(target_ids)
    }


def _format_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _short_target_name(code: str, description: str) -> str:
    joined = f"{code} {description}".upper()
    if "YJ83" in joined or "YJ-83" in joined or "MISSILE" in joined:
        return "YJ-83"
    if "USV" in joined or "UNMANNED SURFACE" in joined:
        return "USV"
    if "UAV" in joined:
        return "UAV"
    if "SHIP" in joined or "DESTROYER" in joined:
        return "Enemy destroyer"
    return description or code


def _target_marker(code: str, description: str) -> str:
    joined = f"{code} {description}".upper()
    if "YJ83" in joined or "YJ-83" in joined or "MISSILE" in joined:
        return "D"
    if "USV" in joined or "UNMANNED SURFACE" in joined:
        return "o"
    if "UAV" in joined:
        return "v"
    if "SHIP" in joined or "DESTROYER" in joined:
        return "s"
    return "o"


def _heading_bearing_deg(vessel: dict[str, Any]) -> float:
    """Vessel heading bearing in degrees clockwise from North."""
    hx = float(vessel.get("HeadingX", 0.0))
    hy = float(vessel.get("HeadingY", 1.0))
    if math.hypot(hx, hy) <= EPS:
        return 0.0
    return math.degrees(math.atan2(hx, hy)) % 360.0


def _bearing_endpoint(
    origin_x: float,
    origin_y: float,
    radius_km: float,
    bearing_deg: float,
) -> tuple[float, float]:
    bearing_rad = math.radians(bearing_deg)
    return (
        origin_x + radius_km * math.sin(bearing_rad),
        origin_y + radius_km * math.cos(bearing_rad),
    )


def _window_links(data: dict[str, Any]) -> list[tuple[int, int, int]]:
    """Return (weapon_id, vessel_id, target_id) for each engagement pair."""
    weapons = data["assignment_request"].get("weapons", [])
    weapon_to_vessel = {
        int(weapon["ID"]): int(weapon["WTAVesselID"])
        for weapon in weapons
    }

    links: list[tuple[int, int, int]] = []
    for key in data.get("engagement_windows", {}):
        weapon_text, target_text = key.split("_", 1)
        weapon_id = int(weapon_text)
        vessel_id = weapon_to_vessel.get(weapon_id)
        if vessel_id is not None:
            links.append((weapon_id, vessel_id, int(target_text)))
    return links


# ---------------------------------------------------------------------------
# Piecewise trajectories
# ---------------------------------------------------------------------------

def _trajectory_points(
    data: dict[str, Any],
    target: dict[str, Any],
    horizon_s: float,
) -> list[dict[str, Any]]:
    """
    Return the target trajectory in local scenario time.

    Real scenarios use `target_trajectories`. Legacy scenarios fall back to
    one straight segment obtained from the target's current velocity.
    """
    target_id = str(target["ID"])
    raw_points = data.get("target_trajectories", {}).get(target_id)

    if raw_points:
        points = sorted(raw_points, key=lambda point: float(point["Time"]))
        clipped: list[dict[str, Any]] = []

        for point in points:
            time_s = float(point["Time"])
            if time_s < -EPS:
                continue
            if time_s <= horizon_s + EPS:
                clipped.append(dict(point))

        # Interpolate a final point when the requested horizon cuts a segment.
        if points and horizon_s < float(points[-1]["Time"]) - EPS:
            final = _position_at_time(points, horizon_s)
            final.update(
                {
                    "Time": horizon_s,
                    "GlobalTime": None,
                    "Waypoint": "horizon",
                }
            )
            if not clipped or abs(float(clipped[-1]["Time"]) - horizon_s) > EPS:
                clipped.append(final)

        if clipped:
            return clipped

    speed_km_s = float(target.get("Speed", 0.0)) * 1e-3
    vx = float(target.get("VX", 0.0)) * speed_km_s
    vy = float(target.get("VY", 0.0)) * speed_km_s
    vz = float(target.get("VZ", 0.0)) * speed_km_s

    x0 = float(target["X"])
    y0 = float(target["Y"])
    z0 = float(target.get("Z", 0.0))

    return [
        {
            "Time": 0.0,
            "Waypoint": "start",
            "X": x0,
            "Y": y0,
            "Z": z0,
        },
        {
            "Time": horizon_s,
            "Waypoint": "linear prediction",
            "X": x0 + vx * horizon_s,
            "Y": y0 + vy * horizon_s,
            "Z": z0 + vz * horizon_s,
        },
    ]


def _position_at_time(
    points: list[dict[str, Any]],
    time_s: float,
) -> dict[str, float]:
    """Linearly interpolate a piecewise route at local time `time_s`."""
    if not points:
        raise ValueError("Cannot interpolate an empty trajectory.")

    ordered = sorted(points, key=lambda point: float(point["Time"]))

    if time_s <= float(ordered[0]["Time"]) + EPS:
        first = ordered[0]
        return {
            "X": float(first["X"]),
            "Y": float(first["Y"]),
            "Z": float(first.get("Z", 0.0)),
        }

    if time_s >= float(ordered[-1]["Time"]) - EPS:
        last = ordered[-1]
        return {
            "X": float(last["X"]),
            "Y": float(last["Y"]),
            "Z": float(last.get("Z", 0.0)),
        }

    for left, right in zip(ordered[:-1], ordered[1:]):
        t0 = float(left["Time"])
        t1 = float(right["Time"])
        if t0 - EPS <= time_s <= t1 + EPS:
            if t1 - t0 <= EPS:
                ratio = 0.0
            else:
                ratio = (time_s - t0) / (t1 - t0)

            return {
                axis: float(left.get(axis, 0.0))
                + ratio * (
                    float(right.get(axis, 0.0))
                    - float(left.get(axis, 0.0))
                )
                for axis in ("X", "Y", "Z")
            }

    last = ordered[-1]
    return {
        "X": float(last["X"]),
        "Y": float(last["Y"]),
        "Z": float(last.get("Z", 0.0)),
    }


def _arrival_time(
    points: list[dict[str, Any]],
    threshold_km: float = 0.05,
) -> float | None:
    """Return the first trajectory-point time at the own ship, when available."""
    for point in points:
        distance = math.sqrt(
            float(point["X"]) ** 2
            + float(point["Y"]) ** 2
            + float(point.get("Z", 0.0)) ** 2
        )
        if distance <= threshold_km:
            return float(point["Time"])
    return None


# ---------------------------------------------------------------------------
# Weapon coverage
# ---------------------------------------------------------------------------

def _draw_weapon_coverage(
    ax: plt.Axes,
    vessel: dict[str, Any],
    weapon: dict[str, Any],
    info: dict[str, Any],
    color: Any,
    display_radius_km: float,
    *,
    fill_alpha: float,
) -> None:
    """
    Draw a weapon's annular azimuth sector.

    Very long ranges are clipped to the current map radius, while the legend
    retains the actual range values.
    """
    x = float(vessel["X"])
    y = float(vessel["Y"])

    min_range_km = max(0.0, float(info.get("MinRange", 0.0)) * 1e-3)
    actual_max_range_km = max(
        min_range_km,
        float(info.get("MaxRange", 0.0)) * 1e-3,
    )
    shown_max_range_km = min(actual_max_range_km, display_radius_km)

    if shown_max_range_km <= EPS:
        return

    heading = _heading_bearing_deg(vessel)
    relative_from = float(info.get("AzimuthFromDeg", 0.0)) % 360.0
    relative_to = float(info.get("AzimuthToDeg", 0.0)) % 360.0

    sweep = (relative_to - relative_from) % 360.0
    if sweep <= EPS:
        sweep = 360.0

    absolute_from = heading + relative_from
    bearings = np.linspace(absolute_from, absolute_from + sweep, 241)

    outer = [
        _bearing_endpoint(x, y, shown_max_range_km, bearing)
        for bearing in bearings
    ]

    if min_range_km > EPS and min_range_km < shown_max_range_km:
        inner = [
            _bearing_endpoint(x, y, min_range_km, bearing)
            for bearing in reversed(bearings)
        ]
        polygon_points = outer + inner
    else:
        polygon_points = [(x, y)] + outer

    # Avoid covering the whole operational plot with the VCM range fill.
    clipped = actual_max_range_km > display_radius_km + EPS
    effective_fill_alpha = 0.0 if clipped else fill_alpha

    if effective_fill_alpha > 0.0:
        polygon = mpatches.Polygon(
            polygon_points,
            closed=True,
            facecolor=(*color[:3], effective_fill_alpha),
            edgecolor="none",
            zorder=1,
        )
        ax.add_patch(polygon)

    outer_x = [point[0] for point in outer]
    outer_y = [point[1] for point in outer]
    ax.plot(
        outer_x,
        outer_y,
        color=color,
        linewidth=1.25,
        linestyle="--" if clipped else "-.",
        alpha=0.8,
        zorder=2,
    )

    if min_range_km > EPS and min_range_km <= display_radius_km:
        inner_points = [
            _bearing_endpoint(x, y, min_range_km, bearing)
            for bearing in bearings
        ]
        ax.plot(
            [point[0] for point in inner_points],
            [point[1] for point in inner_points],
            color=color,
            linewidth=0.9,
            linestyle=":",
            alpha=0.75,
            zorder=2,
        )

    if sweep < 359.999:
        for bearing in (absolute_from, absolute_from + sweep):
            end_x, end_y = _bearing_endpoint(
                x,
                y,
                shown_max_range_km,
                bearing,
            )
            ax.plot(
                [x, end_x],
                [y, end_y],
                color=color,
                linewidth=0.9,
                linestyle=":",
                alpha=0.7,
                zorder=2,
            )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_axis(
    ax: plt.Axes,
    *,
    title: str,
    data: dict[str, Any],
    vessels: list[dict[str, Any]],
    weapons: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    trajectories: dict[int, list[dict[str, Any]]],
    type_info: dict[str, dict[str, Any]],
    weapon_info: dict[str, dict[str, Any]],
    target_colors: dict[int, Any],
    weapon_colors: dict[int, Any],
    horizon_s: float,
    marker_step_s: float,
    arrow_seconds: float,
    show_time_markers: bool,
    show_window_links: bool,
    display_radius_km: float,
    close_in: bool,
) -> None:
    vessels_by_id = {int(vessel["ID"]): vessel for vessel in vessels}
    targets_by_id = {int(target["ID"]): target for target in targets}

    # Weapon coverage is drawn first, behind tactical objects.
    for weapon in weapons:
        vessel = vessels_by_id.get(int(weapon["WTAVesselID"]))
        info = weapon_info.get(str(weapon["WTAWeaponInfoCode"]))
        if vessel is None or info is None:
            continue
        _draw_weapon_coverage(
            ax,
            vessel,
            weapon,
            info,
            weapon_colors[int(weapon["ID"])],
            display_radius_km,
            fill_alpha=0.055 if close_in else 0.025,
        )

    # Own vessel and defense radius.
    for vessel in vessels:
        x = float(vessel["X"])
        y = float(vessel["Y"])
        vessel_id = int(vessel["ID"])

        defense_radius = float(vessel.get("DefenseRadius", 0.0))
        if defense_radius > EPS and defense_radius <= display_radius_km * 1.25:
            circle = plt.Circle(
                (x, y),
                defense_radius,
                facecolor=(0.043, 0.447, 0.522, 0.035),
                edgecolor="#0b7285",
                linewidth=1.0,
                linestyle="--",
                zorder=1,
            )
            ax.add_patch(circle)

        ax.scatter(
            [x],
            [y],
            marker="*",
            s=270,
            c=["#0b7285"],
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
        )
        ax.annotate(
            f"Own ship V{vessel_id}",
            (x, y),
            xytext=(9, -18 if close_in else 9),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            zorder=11,
            arrowprops=(
                dict(arrowstyle="-", color="#0b7285", linewidth=0.7)
                if close_in
                else None
            ),
        )

        heading = _heading_bearing_deg(vessel)
        heading_length = min(2.5, display_radius_km * 0.15)
        heading_x, heading_y = _bearing_endpoint(
            x,
            y,
            heading_length,
            heading,
        )
        ax.annotate(
            "",
            xy=(heading_x, heading_y),
            xytext=(x, y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#0b7285",
                linewidth=1.8,
            ),
            zorder=9,
        )
        ax.text(
            heading_x,
            heading_y,
            f" bow {heading:.1f}°",
            fontsize=7.5,
            color="#0b7285",
            va="bottom",
            zorder=10,
        )

    # Optional engagement links use weapon colors.
    if show_window_links:
        for weapon_id, vessel_id, target_id in _window_links(data):
            vessel = vessels_by_id.get(vessel_id)
            target = targets_by_id.get(target_id)
            if vessel is None or target is None:
                continue

            ax.plot(
                [float(vessel["X"]), float(target["X"])],
                [float(vessel["Y"]), float(target["Y"])],
                color=weapon_colors.get(weapon_id, "0.5"),
                linewidth=0.8,
                linestyle=":",
                alpha=0.22,
                zorder=3,
            )

    for target in sorted(targets, key=lambda row: int(row["ID"])):
        target_id = int(target["ID"])
        code = str(target["WTATargetInfoCode"])
        description = str(type_info.get(code, {}).get("Description", code))
        short_name = _short_target_name(code, description)
        marker = _target_marker(code, description)
        color = target_colors[target_id]
        points = trajectories[target_id]

        xs = [float(point["X"]) for point in points]
        ys = [float(point["Y"]) for point in points]

        # Piecewise route.
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.7,
            linestyle="--",
            alpha=0.78,
            zorder=5,
        )

        # Current position at local t=0.
        x0 = float(target["X"])
        y0 = float(target["Y"])
        ax.scatter(
            [x0],
            [y0],
            marker=marker,
            s=125 if marker != "s" else 145,
            c=[color],
            edgecolors="black",
            linewidths=0.7,
            zorder=8,
        )

        close_offsets = {
            1: (7, 8),
            2: (8, 10),
            3: (-58, 12),
            4: (-56, -18),
            5: (8, -18),
        }
        label_offset = (
            close_offsets.get(target_id, (6, 6))
            if close_in
            else (6, 7)
        )

        ax.annotate(
            f"T{target_id} {short_name}",
            (x0, y0),
            xytext=label_offset,
            textcoords="offset points",
            fontsize=8.3,
            fontweight="bold",
            color=color,
            zorder=9,
            arrowprops=(
                dict(arrowstyle="-", color=color, linewidth=0.65, alpha=0.8)
                if close_in
                else None
            ),
        )

        # Current velocity arrow.
        speed_km_s = float(target.get("Speed", 0.0)) * 1e-3
        dx = float(target.get("VX", 0.0)) * speed_km_s * arrow_seconds
        dy = float(target.get("VY", 0.0)) * speed_km_s * arrow_seconds
        ax.arrow(
            x0,
            y0,
            dx,
            dy,
            length_includes_head=True,
            head_width=max(0.06, 0.007 * display_radius_km),
            head_length=max(0.10, 0.012 * display_radius_km),
            linewidth=1.2,
            color=color,
            alpha=0.9,
            zorder=7,
        )

        # Explicit route waypoints/turns from the JSON.
        for index, point in enumerate(points[1:], start=1):
            point_time = float(point["Time"])
            point_x = float(point["X"])
            point_y = float(point["Y"])
            waypoint = str(point.get("Waypoint", ""))

            ax.scatter(
                [point_x],
                [point_y],
                marker="x" if index == len(points) - 1 else "o",
                s=45 if index == len(points) - 1 else 28,
                c=[color],
                linewidths=1.2,
                zorder=7,
            )

            point_distance = math.hypot(point_x, point_y)
            if (
                waypoint
                and waypoint not in {"horizon", "linear prediction"}
                and point_distance > 0.25
            ):
                ax.annotate(
                    f"{waypoint}\nt={point_time:.1f}s",
                    (point_x, point_y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6.8,
                    color=color,
                    zorder=8,
                )

        # Regular local-time markers along the actual piecewise route.
        if show_time_markers and marker_step_s > EPS:
            final_time = min(horizon_s, float(points[-1]["Time"]))
            marker_time = marker_step_s
            while marker_time < final_time - EPS:
                position = _position_at_time(points, marker_time)
                ax.scatter(
                    [position["X"]],
                    [position["Y"]],
                    marker=".",
                    s=20,
                    c=[color],
                    alpha=0.62,
                    zorder=6,
                )
                label_period = max(marker_step_s * 2.0, 1.0)
                if abs((marker_time / label_period) - round(marker_time / label_period)) <= 1e-6:
                    ax.annotate(
                        f"{marker_time:g}s",
                        (position["X"], position["Y"]),
                        xytext=(3, 2),
                        textcoords="offset points",
                        fontsize=6.2,
                        color=color,
                        alpha=0.8,
                        zorder=7,
                    )
                marker_time += marker_step_s

    ax.axhline(0.0, color="gray", linewidth=0.7, alpha=0.55)
    ax.axvline(0.0, color="gray", linewidth=0.7, alpha=0.55)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (km, East)")
    ax.set_ylabel("Y (km, North)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")


def plot_scenario(
    scenario_path: str,
    out_path: str | None = None,
    *,
    trajectory_seconds: float | None = None,
    marker_step: float = 15.0,
    arrow_seconds: float = 5.0,
    pad_km: float = 3.0,
    close_range_km: float | None = None,
    show_time_markers: bool = True,
    show_window_links: bool = True,
) -> None:
    with open(scenario_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    request = data["assignment_request"]
    vessels = request.get("vessels", [])
    weapons = request.get("weapons", [])
    targets = request.get("targets", [])

    if not vessels:
        raise ValueError("No own vessel was found in the scenario.")
    if not targets:
        raise ValueError("No targets were found in the scenario.")

    metadata = data.get("scenario_metadata", {})
    scenario_horizon = float(
        metadata.get(
            "scenario_horizon_s",
            max(
                (
                    float(interval[1])
                    for interval in data.get("engagement_windows", {}).values()
                ),
                default=60.0,
            ),
        )
    )

    horizon_s = (
        scenario_horizon
        if trajectory_seconds is None
        else min(float(trajectory_seconds), scenario_horizon)
    )

    type_info = _target_type_lookup(data)
    weapon_info = _weapon_info_lookup(data)
    target_colors = _target_color_map(targets)

    weapon_palette = matplotlib.colormaps["Dark2"]
    sorted_weapons = sorted(weapons, key=lambda row: int(row["ID"]))
    weapon_colors = {
        int(weapon["ID"]): weapon_palette(
            index / max(len(sorted_weapons) - 1, 1)
        )
        for index, weapon in enumerate(sorted_weapons)
    }

    trajectories = {
        int(target["ID"]): _trajectory_points(data, target, horizon_s)
        for target in targets
    }

    all_points = [
        (
            float(point["X"]),
            float(point["Y"]),
        )
        for points in trajectories.values()
        for point in points
    ]
    all_points.extend(
        (float(vessel["X"]), float(vessel["Y"]))
        for vessel in vessels
    )

    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    span_x = x_max - x_min
    span_y = y_max - y_min
    overview_half_span = max(span_x, span_y, 1.0) / 2.0 + pad_km
    overview_center_x = (x_min + x_max) / 2.0
    overview_center_y = (y_min + y_max) / 2.0

    if close_range_km is None:
        close_weapon_ranges = [
            float(info.get("MaxRange", 0.0)) * 1e-3
            for info in weapon_info.values()
            if float(info.get("MaxRange", 0.0)) * 1e-3 <= 30.0
        ]
        close_target_distances = [
            math.hypot(float(target["X"]), float(target["Y"]))
            for target in targets
            if "SHIP" not in str(target["WTATargetInfoCode"]).upper()
        ]
        close_range_km = max(
            12.0,
            max(close_weapon_ranges, default=15.0) * 1.1,
            max(close_target_distances, default=10.0) + 2.0,
        )

    fig, (overview_ax, close_ax) = plt.subplots(
        1,
        2,
        figsize=(19, 9),
        gridspec_kw={"width_ratios": [1.22, 1.0]},
    )

    _draw_axis(
        overview_ax,
        title="Operational overview",
        data=data,
        vessels=vessels,
        weapons=weapons,
        targets=targets,
        trajectories=trajectories,
        type_info=type_info,
        weapon_info=weapon_info,
        target_colors=target_colors,
        weapon_colors=weapon_colors,
        horizon_s=horizon_s,
        marker_step_s=marker_step,
        arrow_seconds=arrow_seconds,
        show_time_markers=show_time_markers,
        show_window_links=show_window_links,
        display_radius_km=overview_half_span * 1.05,
        close_in=False,
    )
    overview_ax.set_xlim(
        overview_center_x - overview_half_span,
        overview_center_x + overview_half_span,
    )
    overview_ax.set_ylim(
        overview_center_y - overview_half_span,
        overview_center_y + overview_half_span,
    )

    _draw_axis(
        close_ax,
        title=f"Close-in defense view (±{close_range_km:g} km)",
        data=data,
        vessels=vessels,
        weapons=weapons,
        targets=targets,
        trajectories=trajectories,
        type_info=type_info,
        weapon_info=weapon_info,
        target_colors=target_colors,
        weapon_colors=weapon_colors,
        horizon_s=horizon_s,
        marker_step_s=marker_step,
        arrow_seconds=arrow_seconds,
        show_time_markers=show_time_markers,
        show_window_links=show_window_links,
        display_radius_km=close_range_km,
        close_in=True,
    )
    close_ax.set_xlim(-close_range_km, close_range_km)
    close_ax.set_ylim(-close_range_km, close_range_km)

    global_start = str(metadata.get("global_start_time", "local t=0"))
    global_end = str(
        metadata.get(
            "global_end_time",
            f"local t={horizon_s:g}s",
        )
    )
    fig.suptitle(
        "Real WTA scenario "
        f"{global_start}–{global_end} "
        f"(local t=0–{horizon_s:g}s)",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------
    legend_handles: list[Any] = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#0b7285",
            markeredgecolor="black",
            markersize=12,
            label=f"Own ship ({len(weapons)} weapons)",
        )
    ]

    for target in sorted(targets, key=lambda row: int(row["ID"])):
        target_id = int(target["ID"])
        code = str(target["WTATargetInfoCode"])
        description = str(type_info.get(code, {}).get("Description", code))
        short_name = _short_target_name(code, description)
        points = trajectories[target_id]
        arrival = _arrival_time(points)
        speed = float(target.get("Speed", 0.0))

        timing_text = (
            f", arrival t={arrival:.1f}s"
            if arrival is not None
            else f", shown through t={float(points[-1]['Time']):.1f}s"
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=_target_marker(code, description),
                color=target_colors[target_id],
                markerfacecolor=target_colors[target_id],
                markeredgecolor="black",
                linestyle="--",
                linewidth=1.4,
                markersize=8,
                label=(
                    f"T{target_id} {short_name}: "
                    f"{speed:.1f} m/s{timing_text}"
                ),
            )
        )

    for weapon in sorted_weapons:
        weapon_id = int(weapon["ID"])
        info = weapon_info.get(str(weapon["WTAWeaponInfoCode"]), {})
        min_range = float(info.get("MinRange", 0.0)) * 1e-3
        max_range = float(info.get("MaxRange", 0.0)) * 1e-3
        azimuth_from = float(info.get("AzimuthFromDeg", 0.0)) % 360.0
        azimuth_to = float(info.get("AzimuthToDeg", 0.0)) % 360.0

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=weapon_colors[weapon_id],
                linestyle="-.",
                linewidth=1.8,
                label=(
                    f"W{weapon_id} {weapon['WTAWeaponInfoCode']}: "
                    f"{min_range:g}–{max_range:g} km, "
                    f"az. {azimuth_from:g}°→{azimuth_to:g}°"
                ),
            )
        )

    figure_legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=2,
        fontsize=8.2,
        title="Targets and own-ship weapons",
        title_fontsize=9,
        framealpha=0.92,
    )

    fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.955))

    if out_path:
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output,
            dpi=180,
            bbox_inches="tight",
            bbox_extra_artists=(figure_legend,),
        )
        print(f"Saved scenario map to {output}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw a WTA scenario with piecewise real trajectories."
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="real_scenario_013000_013230.json",
        help="Path to the WTA scenario JSON.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Save the figure instead of opening it interactively.",
    )
    parser.add_argument(
        "--trajectory-seconds",
        type=float,
        default=None,
        help=(
            "Local trajectory duration to draw. By default, use "
            "scenario_metadata.scenario_horizon_s."
        ),
    )
    parser.add_argument(
        "--marker-step",
        type=float,
        default=15.0,
        help="Spacing between local-time trajectory markers in seconds.",
    )
    parser.add_argument(
        "--arrow-seconds",
        type=float,
        default=5.0,
        help="Length of the current-velocity arrow measured in travel seconds.",
    )
    parser.add_argument(
        "--pad-km",
        type=float,
        default=3.0,
        help="Padding around the operational overview.",
    )
    parser.add_argument(
        "--close-range-km",
        type=float,
        default=None,
        help="Half-width of the close-in panel. Default is selected automatically.",
    )
    parser.add_argument(
        "--no-time-markers",
        action="store_true",
        help="Hide regular time markers along target routes.",
    )
    parser.add_argument(
        "--no-window-links",
        action="store_true",
        help="Hide weapon-target engagement links.",
    )

    args = parser.parse_args()

    if args.trajectory_seconds is not None and args.trajectory_seconds <= 0:
        raise ValueError("--trajectory-seconds must be positive.")
    if args.marker_step <= 0:
        raise ValueError("--marker-step must be positive.")
    if args.arrow_seconds <= 0:
        raise ValueError("--arrow-seconds must be positive.")
    if args.pad_km < 0:
        raise ValueError("--pad-km must be non-negative.")
    if args.close_range_km is not None and args.close_range_km <= 0:
        raise ValueError("--close-range-km must be positive.")

    plot_scenario(
        scenario_path=args.scenario,
        out_path=args.out,
        trajectory_seconds=args.trajectory_seconds,
        marker_step=args.marker_step,
        arrow_seconds=args.arrow_seconds,
        pad_km=args.pad_km,
        close_range_km=args.close_range_km,
        show_time_markers=not args.no_time_markers,
        show_window_links=not args.no_window_links,
    )


if __name__ == "__main__":
    main()
