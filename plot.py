#!/usr/bin/env python3
"""
plot.py — Visualise a WTA solution as a Gantt chart.

Two subplots:
  Top   : Weapon schedule — each weapon is a row; each burst is a coloured bar;
          colour encodes target ID.  Engagement windows shown as thin outlines.
  Bottom: Target survival — bar chart of per-target residual threat w_j * Π(1-p)^k.

Usage:
    python plot.py [scenario.json] [solution.json] [--out figure.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(scenario_path: str, solution_path: str):
    with open(scenario_path) as f:
        sc = json.load(f)
    with open(solution_path) as f:
        sol = json.load(f)

    weapon_info = {d["Code"]: d for d in sc["weapon_infos"]}
    req = sc["assignment_request"]

    weapons = {}
    for d in req["weapons"]:
        d = dict(d)
        wi = weapon_info[d["WTAWeaponInfoCode"]]
        d["_burst_dur"] = wi["BurstInterval"] + wi["ReloadTime"]
        weapons[d["ID"]] = d

    targets = {d["ID"]: d for d in req["targets"]}

    prob: Dict[Tuple[str, str], float] = {}
    for row in sc["probability_table"]:
        prob[(row["WTAWeaponInfoCode"], row["WTATargetInfoCode"])] = row["Score"]

    windows: Dict[Tuple[int, int], Tuple[float, float]] = {}
    p_ij: Dict[Tuple[int, int], float] = {}
    for key_str, (a, b) in sc["engagement_windows"].items():
        wid, tid = map(int, key_str.split("_"))
        if wid not in weapons or tid not in targets:
            continue
        wcode = weapons[wid]["WTAWeaponInfoCode"]
        tcode = targets[tid]["WTATargetInfoCode"]
        p = prob.get((wcode, tcode), 0.0)
        if p > 0.0:
            windows[(wid, tid)] = (a, b)
            p_ij[(wid, tid)] = p

    assignments = sol.get("assignments", [])
    objective = sol.get("objective")
    horizon = max(b for _, b in windows.values()) if windows else 60.0

    return weapons, targets, windows, p_ij, assignments, objective, horizon


# ---------------------------------------------------------------------------
# Build per-target survival
# ---------------------------------------------------------------------------

def compute_survival(targets, p_ij, assignments):
    survival = {tid: 1.0 for tid in targets}
    for a in assignments:
        wid, tid, ammo = a["WTAWeaponID"], a["WTATargetID"], a["AmmoUsed"]
        p = p_ij.get((wid, tid), 0.0)
        survival[tid] *= (1.0 - p) ** ammo
    threat = {tid: targets[tid]["ThreatScore"] * survival[tid] for tid in targets}
    return survival, threat


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(scenario_path: str, solution_path: str, out_path: str | None = None):
    weapons, targets, windows, p_ij, assignments, objective, horizon = \
        load(scenario_path, solution_path)

    survival, threat = compute_survival(targets, p_ij, assignments)

    # --- colour map: one colour per target ID ---
    all_tids = sorted(targets.keys())
    cmap = matplotlib.colormaps["tab20"] if len(all_tids) <= 20 else matplotlib.colormaps["hsv"]
    tid_color = {tid: cmap(i / max(len(all_tids) - 1, 1)) for i, tid in enumerate(all_tids)}

    # --- weapon ordering: group by vessel ---
    weapon_ids = sorted(weapons.keys(), key=lambda w: (weapons[w]["WTAVesselID"], w))
    wid_to_row = {wid: i for i, wid in enumerate(weapon_ids)}
    n_weapons  = len(weapon_ids)

    # Separate weapons that appear in the solution
    active_wids = {a["WTAWeaponID"] for a in assignments}

    fig, (ax_gantt, ax_surv) = plt.subplots(
        2, 1,
        figsize=(16, max(8, 0.35 * n_weapons) + 4),
        gridspec_kw={"height_ratios": [max(4, 0.35 * n_weapons), 3]},
    )

    # ------------------------------------------------------------------ Gantt
    bh = 0.6   # bar half-height

    # Draw engagement window outlines (light, behind burst bars)
    drawn_windows: set = set()
    for (wid, tid), (a_win, b_win) in windows.items():
        if wid not in wid_to_row:
            continue
        row = wid_to_row[wid]
        color = tid_color[tid]
        rect = mpatches.FancyArrowPatch(
            posA=(a_win, row - bh * 0.5),
            posB=(b_win, row - bh * 0.5),
            arrowstyle="-",
            color=(*color[:3], 0.18),
            linewidth=6,
        )
        # Simpler: just a thin underline
        ax_gantt.plot([a_win, b_win], [row - bh * 0.75, row - bh * 0.75],
                      color=(*color[:3], 0.25), linewidth=2, solid_capstyle="butt")

    # Draw burst bars
    for a in assignments:
        wid  = a["WTAWeaponID"]
        tid  = a["WTATargetID"]
        ammo = a["AmmoUsed"]
        d    = weapons[wid]["_burst_dur"]
        color = tid_color[tid]
        row   = wid_to_row[wid]
        fire_times: List[float] = a.get("FireTimes") or [a["FireTime"]]

        for ft in fire_times:
            rect = mpatches.Rectangle(
                (ft, row - bh / 2), d, bh,
                facecolor=color, edgecolor="black", linewidth=0.4, alpha=0.85,
            )
            ax_gantt.add_patch(rect)
            # Label target inside bar if wide enough
            if d > horizon * 0.012:
                ax_gantt.text(
                    ft + d / 2, row, str(tid),
                    ha="center", va="center", fontsize=5, color="white",
                    fontweight="bold", clip_on=True,
                )

    ax_gantt.set_xlim(0, horizon)
    ax_gantt.set_ylim(-0.8, n_weapons - 0.2)
    ax_gantt.set_yticks(range(n_weapons))
    ax_gantt.set_yticklabels(
        [f"W{wid} (V{weapons[wid]['WTAVesselID']})" for wid in weapon_ids],
        fontsize=7,
    )
    ax_gantt.set_xlabel("Time (s)")
    ax_gantt.set_title(
        f"Weapon Schedule — {len(assignments)} assignments  |  objective = {objective:.4f}",
        fontsize=11,
    )
    ax_gantt.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5)
    # Vessel boundary lines
    prev_vessel = None
    for i, wid in enumerate(weapon_ids):
        v = weapons[wid]["WTAVesselID"]
        if prev_vessel is not None and v != prev_vessel:
            ax_gantt.axhline(i - 0.5, color="navy", linewidth=1.2, linestyle="-")
        prev_vessel = v

    # Legend (up to 30 targets)
    legend_tids = sorted({a["WTATargetID"] for a in assignments})
    patches = [mpatches.Patch(color=tid_color[t], label=f"T{t}") for t in legend_tids[:30]]
    ax_gantt.legend(handles=patches, loc="upper right", ncol=6,
                    fontsize=6, title="Target", title_fontsize=7,
                    framealpha=0.8)

    # -------------------------------------------------------- Target survival
    sorted_tids = sorted(targets.keys(), key=lambda t: -targets[t]["ThreatScore"])
    threat_vals  = [threat[t] for t in sorted_tids]
    initial_vals = [targets[t]["ThreatScore"] for t in sorted_tids]
    colors_surv  = [tid_color[t] for t in sorted_tids]

    x = np.arange(len(sorted_tids))
    ax_surv.bar(x, initial_vals, width=0.8, color=[(*c[:3], 0.25) for c in colors_surv],
                edgecolor="none", label="Initial threat")
    ax_surv.bar(x, threat_vals, width=0.8, color=[(*c[:3], 0.85) for c in colors_surv],
                edgecolor="none", label="Residual threat")

    ax_surv.set_xticks(x)
    ax_surv.set_xticklabels([str(t) for t in sorted_tids], fontsize=6, rotation=90)
    ax_surv.set_ylabel("Threat score")
    ax_surv.set_xlabel("Target ID (sorted by initial threat)")
    ax_surv.set_title("Residual Threat per Target (dark = remaining, light = neutralised)")
    ax_surv.legend(fontsize=8)
    ax_surv.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    plt.tight_layout(pad=1.5)

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot WTA solution Gantt chart")
    parser.add_argument("scenario", nargs="?",
                        default="/workspaces/WTA/data/scenario_001.json")
    parser.add_argument("solution", nargs="?",
                        default="/workspaces/WTA/data/scenario_001_solution.json")
    parser.add_argument("--out", default=None,
                        help="Save figure to this path instead of showing (e.g. plot.png)")
    args = parser.parse_args()
    plot(args.scenario, args.solution, args.out)
