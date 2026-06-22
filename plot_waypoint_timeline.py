#!/usr/bin/env python3
"""
Plot target waypoint routes and calculate arrival times from scenario(1).xlsx.

Outputs:
  - waypoint_timeline.csv
  - waypoint_routes.png
  - distance_to_own_ship.png

Assumptions:
  1. Each target moves at constant speed between consecutive waypoints.
  2. D0 occurs at t = 0 for every target.
  3. "Tàu khu trục D0" refers to the enemy destroyer's D0 coordinate.
  4. "Tàu ta" refers to the own-vessel coordinate.
  5. Longitude W is negative and latitude N is positive.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from openpyxl import load_workbook


EARTH_RADIUS_KM = 6371.0088
WAYPOINT_COLUMNS = {
    "D0": 4,
    "D1": 5,
    "D2": 6,
    "D3": 7,
    "D4": 8,
}


def parse_coordinate(value: object) -> Optional[Tuple[float, float]]:
    """Parse strings such as '109.510793°W / 11.951853°N' as (lat, lon)."""
    if not isinstance(value, str):
        return None

    pattern = (
        r"([0-9]+(?:\.[0-9]+)?)\s*°?\s*([EW])"
        r"\s*/\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*°?\s*([NS])"
    )
    match = re.search(pattern, value.strip(), flags=re.IGNORECASE)
    if not match:
        return None

    lon = float(match.group(1))
    if match.group(2).upper() == "W":
        lon = -lon

    lat = float(match.group(3))
    if match.group(4).upper() == "S":
        lat = -lat

    return lat, lon


def parse_speed_kmh(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"Cannot parse speed from {value!r}")

    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", value)
    if not match:
        raise ValueError(f"Cannot parse speed from {value!r}")
    return float(match.group())


def parse_altitude_m(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return 0.0

    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", value)
    return float(match.group()) if match else 0.0


def latlon_to_local_km(
    lat: float,
    lon: float,
    origin_lat: float,
    origin_lon: float,
) -> Tuple[float, float]:
    """
    Convert latitude/longitude to local Cartesian coordinates.

    X is eastward displacement in km.
    Y is northward displacement in km.
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(origin_lat)
    lon0_rad = math.radians(origin_lon)

    x = EARTH_RADIUS_KM * math.cos(lat0_rad) * (lon_rad - lon0_rad)
    y = EARTH_RADIUS_KM * (lat_rad - lat0_rad)
    return x, y


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Great-circle surface distance in kilometres."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def distance_3d_km(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
) -> float:
    """3-D distance using haversine horizontal distance and altitude difference."""
    lat1, lon1, alt1_m = p1
    lat2, lon2, alt2_m = p2
    horizontal = haversine_km(lat1, lon1, lat2, lon2)
    vertical = (alt2_m - alt1_m) / 1000.0
    return math.hypot(horizontal, vertical)


def format_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def load_routes(
    workbook_path: Path,
) -> Tuple[Tuple[float, float], List[dict]]:
    wb = load_workbook(workbook_path, data_only=True)
    ws = wb["Sheet1"]

    # The detailed scenario table starts at row 30.
    own_name = str(ws.cell(30, 1).value).strip()
    own_coord = parse_coordinate(ws.cell(30, 4).value)
    if own_coord is None:
        raise ValueError("Could not parse the own-vessel coordinate from cell D30.")

    # Resolve the enemy destroyer's D0 for all targets that reference it.
    enemy_d0 = parse_coordinate(ws.cell(31, 4).value)
    if enemy_d0 is None:
        raise ValueError("Could not parse the enemy destroyer's D0 coordinate from D31.")

    references: Dict[str, Tuple[float, float]] = {
        "Tàu ta": own_coord,
        "Tau ta": own_coord,
        "Tàu khu trục D0": enemy_d0,
        "Tau khu truc D0": enemy_d0,
    }

    routes: List[dict] = []

    for row in range(31, ws.max_row + 1):
        name_value = ws.cell(row, 1).value
        if not name_value:
            continue

        name = str(name_value).strip()
        speed_kmh = parse_speed_kmh(ws.cell(row, 2).value)
        altitude_m = parse_altitude_m(ws.cell(row, 3).value)

        raw_points: List[Tuple[str, float, float, float]] = []
        for label, column in WAYPOINT_COLUMNS.items():
            raw = ws.cell(row, column).value
            if raw is None or str(raw).strip() in {"", "-"}:
                continue

            coord = parse_coordinate(raw)
            if coord is None and isinstance(raw, str):
                coord = references.get(raw.strip())

            if coord is None:
                raise ValueError(
                    f"Unrecognized waypoint value for {name} {label}: {raw!r}"
                )

            lat, lon = coord
            raw_points.append((label, lat, lon, altitude_m))

        if len(raw_points) < 2:
            continue

        points = []
        elapsed_s = 0.0
        cumulative_distance_km = 0.0

        for index, (label, lat, lon, alt_m) in enumerate(raw_points):
            if index > 0:
                prev = raw_points[index - 1]
                segment_distance_km = distance_3d_km(
                    (prev[1], prev[2], prev[3]),
                    (lat, lon, alt_m),
                )
                segment_time_s = segment_distance_km / speed_kmh * 3600.0
                elapsed_s += segment_time_s
                cumulative_distance_km += segment_distance_km
            else:
                segment_distance_km = 0.0
                segment_time_s = 0.0

            x_km, y_km = latlon_to_local_km(
                lat, lon, own_coord[0], own_coord[1]
            )

            points.append(
                {
                    "label": label,
                    "lat": lat,
                    "lon": lon,
                    "altitude_m": alt_m,
                    "x_km": x_km,
                    "y_km": y_km,
                    "segment_distance_km": segment_distance_km,
                    "segment_time_s": segment_time_s,
                    "cumulative_distance_km": cumulative_distance_km,
                    "arrival_time_s": elapsed_s,
                }
            )

        routes.append(
            {
                "name": name,
                "speed_kmh": speed_kmh,
                "altitude_m": altitude_m,
                "points": points,
            }
        )

    if not routes:
        raise ValueError("No target routes were found in the workbook.")

    return own_coord, routes


def write_timeline_csv(routes: List[dict], output_path: Path) -> None:
    fields = [
        "target",
        "speed_kmh",
        "waypoint",
        "latitude_deg",
        "longitude_deg",
        "altitude_m",
        "x_east_km",
        "y_north_km",
        "segment_distance_km",
        "segment_time_s",
        "cumulative_distance_km",
        "arrival_time_s",
        "arrival_time_hms",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for route in routes:
            for point in route["points"]:
                writer.writerow(
                    {
                        "target": route["name"],
                        "speed_kmh": round(route["speed_kmh"], 3),
                        "waypoint": point["label"],
                        "latitude_deg": round(point["lat"], 8),
                        "longitude_deg": round(point["lon"], 8),
                        "altitude_m": round(point["altitude_m"], 3),
                        "x_east_km": round(point["x_km"], 4),
                        "y_north_km": round(point["y_km"], 4),
                        "segment_distance_km": round(
                            point["segment_distance_km"], 4
                        ),
                        "segment_time_s": round(point["segment_time_s"], 3),
                        "cumulative_distance_km": round(
                            point["cumulative_distance_km"], 4
                        ),
                        "arrival_time_s": round(point["arrival_time_s"], 3),
                        "arrival_time_hms": format_elapsed(
                            point["arrival_time_s"]
                        ),
                    }
                )


def plot_routes(routes: List[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 9))

    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=220,
        label="Own vessel",
        zorder=10,
    )
    ax.annotate(
        "Own vessel\n(0, 0)",
        (0.0, 0.0),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
    )

    for route in routes:
        xs = [p["x_km"] for p in route["points"]]
        ys = [p["y_km"] for p in route["points"]]

        line = ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=f'{route["name"]} ({route["speed_kmh"]:g} km/h)',
        )[0]

        for point in route["points"]:
            annotation = (
                f'{point["label"]}\n'
                f't={format_elapsed(point["arrival_time_s"])}'
            )
            ax.annotate(
                annotation,
                (point["x_km"], point["y_km"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color=line.get_color(),
            )

    ax.set_title("Target waypoint routes and calculated arrival times")
    ax.set_xlabel("East displacement from own vessel (km)")
    ax.set_ylabel("North displacement from own vessel (km)")
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_distance_to_ship(routes: List[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    for route in routes:
        times_min = [p["arrival_time_s"] / 60.0 for p in route["points"]]
        distances = [
            math.sqrt(
                p["x_km"] ** 2
                + p["y_km"] ** 2
                + (p["altitude_m"] / 1000.0) ** 2
            )
            for p in route["points"]
        ]

        line = ax.plot(
            times_min,
            distances,
            marker="o",
            linewidth=1.8,
            label=route["name"],
        )[0]

        for point, time_min, distance in zip(
            route["points"], times_min, distances
        ):
            ax.annotate(
                point["label"],
                (time_min, distance),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=line.get_color(),
            )

    ax.set_title("Target distance to own vessel at each waypoint")
    ax.set_xlabel("Elapsed time from D0 (minutes)")
    ax.set_ylabel("Distance to own vessel (km)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate waypoint arrival times and plot target routes."
    )
    parser.add_argument(
        "workbook",
        nargs="?",
        default="scenario(1).xlsx",
        help="Path to the scenario Excel file.",
    )
    parser.add_argument(
        "--outdir",
        default="waypoint_output",
        help="Directory for generated CSV and PNG files.",
    )
    args = parser.parse_args()

    workbook_path = Path(args.workbook)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, routes = load_routes(workbook_path)

    csv_path = output_dir / "waypoint_timeline.csv"
    route_plot_path = output_dir / "waypoint_routes.png"
    distance_plot_path = output_dir / "distance_to_own_ship.png"

    write_timeline_csv(routes, csv_path)
    plot_routes(routes, route_plot_path)
    plot_distance_to_ship(routes, distance_plot_path)

    print(f"Created: {csv_path}")
    print(f"Created: {route_plot_path}")
    print(f"Created: {distance_plot_path}")

    for route in routes:
        final = route["points"][-1]
        print(
            f'{route["name"]}: '
            f'{final["cumulative_distance_km"]:.2f} km, '
            f'{format_elapsed(final["arrival_time_s"])}'
        )


if __name__ == "__main__":
    main()
