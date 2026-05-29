#!/usr/bin/env python3
"""
eda.py — Exploratory Data Analysis for a WTA scenario JSON.

Produces a multi-page PDF (or PNG per figure) with:
  1. Weapon overview    — ammo dist, burst_dur dist, type breakdown, ammo by vessel
  2. Target overview    — threat score dist, speed dist, type breakdown, spatial XY
  3. Engagement network — weapons-per-target, targets-per-weapon, p_kill dist
  4. Window analysis    — window width dist, feasible slots per pair, coverage heatmap
  5. Pair-level detail  — scatter: window_width vs p_kill, ammo vs targets-per-weapon

Usage:
    python eda.py [scenario.json] [--out eda.pdf]
    python eda.py [scenario.json] --out eda.png   # saves eda_fig1.png etc.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import product
from math import floor
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 9,
                             "xtick.labelsize": 8, "ytick.labelsize": 8})

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(path: str):
    with open(path) as f:
        data = json.load(f)

    winfo = {d["Code"]: d for d in data["weapon_infos"]}
    tinfo = {d["Code"]: d for d in data["target_infos"]}
    prob  = {(r["WTAWeaponInfoCode"], r["WTATargetInfoCode"]): r["Score"]
             for r in data["probability_table"]}

    req = data["assignment_request"]

    weapons = {}
    for d in req["weapons"]:
        d = dict(d)
        wi = winfo[d["WTAWeaponInfoCode"]]
        d.update({
            "_burst_dur": wi["BurstInterval"] + wi["ReloadTime"],
            "_M":         wi["MaxShotsPerTarget"],
            "_type":      wi["Type"],
            "_info":      wi,
        })
        weapons[d["ID"]] = d

    targets = {}
    for d in req["targets"]:
        d = dict(d)
        ti = tinfo[d["WTATargetInfoCode"]]
        d["_type"] = ti["Type"]
        d["_desc"] = ti.get("Description", d["WTATargetInfoCode"])
        targets[d["ID"]] = d

    windows: Dict[Tuple[int, int], Tuple[float, float]] = {}
    p_ij:    Dict[Tuple[int, int], float] = {}
    for key_str, (a, b) in data["engagement_windows"].items():
        wid, tid = map(int, key_str.split("_"))
        if wid not in weapons or tid not in targets:
            continue
        p = prob.get((weapons[wid]["WTAWeaponInfoCode"],
                       targets[tid]["WTATargetInfoCode"]), 0.0)
        if p > 0:
            windows[(wid, tid)] = (a, b)
            p_ij[(wid, tid)] = p

    return weapons, targets, windows, p_ij


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def derive(weapons, targets, windows, p_ij):
    weapons_per_target = defaultdict(list)
    targets_per_weapon = defaultdict(list)
    window_widths:  List[float] = []
    feasible_slots: List[int]   = []
    p_vals:         List[float] = []

    for (wid, tid), (a, b) in windows.items():
        d = weapons[wid]["_burst_dur"]
        width = b - a
        slots = int(width / d) if width >= d else 0
        window_widths.append(width)
        feasible_slots.append(slots)
        p_vals.append(p_ij[(wid, tid)])
        weapons_per_target[tid].append(wid)
        targets_per_weapon[wid].append(tid)

    return weapons_per_target, targets_per_weapon, window_widths, feasible_slots, p_vals


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def fig_weapons(weapons):
    wids  = sorted(weapons)
    ammos = [weapons[w]["Ammo"] for w in wids]
    durs  = [weapons[w]["_burst_dur"] for w in wids]
    types = [weapons[w]["_type"] for w in wids]
    vessels = sorted({weapons[w]["WTAVesselID"] for w in wids})

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Weapon Overview", fontsize=12, fontweight="bold")

    # ammo distribution
    ax = axes[0, 0]
    ax.hist(ammos, bins=30, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(ammos), color="red", linestyle="--", linewidth=1, label=f"mean={np.mean(ammos):.0f}")
    ax.set_title("Ammo Distribution")
    ax.set_xlabel("Ammo capacity")
    ax.set_ylabel("# Weapons")
    ax.legend(fontsize=8)

    # burst duration distribution
    ax = axes[0, 1]
    unique_durs, counts = np.unique(durs, return_counts=True)
    ax.bar(unique_durs, counts, width=np.min(np.diff(unique_durs)) * 0.5 if len(unique_durs) > 1 else 1,
           color="darkorange", edgecolor="white")
    ax.set_title("Burst Duration Distribution")
    ax.set_xlabel("Burst duration (BurstInterval + ReloadTime) [s]")
    ax.set_ylabel("# Weapons")
    for x, c in zip(unique_durs, counts):
        ax.text(x, c + 0.3, str(c), ha="center", fontsize=8)

    # weapon type breakdown
    ax = axes[1, 0]
    type_counts = defaultdict(int)
    type_names  = defaultdict(str)
    for w in weapons.values():
        wi = w["_info"]
        type_counts[wi["Code"]] += 1
    codes = sorted(type_counts)
    vals  = [type_counts[c] for c in codes]
    colors = plt.cm.Set2(np.linspace(0, 1, len(codes)))
    bars = ax.barh(codes, vals, color=colors)
    ax.set_title("Weapon Types (by WTAWeaponInfoCode)")
    ax.set_xlabel("# Weapons")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.2, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=8)

    # ammo by vessel
    ax = axes[1, 1]
    vessel_ammo = defaultdict(list)
    for w in weapons.values():
        vessel_ammo[w["WTAVesselID"]].append(w["Ammo"])
    vs = sorted(vessel_ammo)
    bp = ax.boxplot([vessel_ammo[v] for v in vs], labels=[f"V{v}" for v in vs],
                    patch_artist=True, medianprops={"color": "red"})
    for patch, c in zip(bp["boxes"], plt.cm.Pastel1(np.linspace(0, 1, len(vs)))):
        patch.set_facecolor(c)
    ax.set_title("Ammo Distribution by Vessel")
    ax.set_xlabel("Vessel")
    ax.set_ylabel("Ammo")

    plt.tight_layout()
    return fig


def fig_targets(targets):
    tids    = sorted(targets)
    threats = [targets[t]["ThreatScore"] for t in tids]
    speeds  = [targets[t]["Speed"] for t in tids]
    xs      = [targets[t]["X"] for t in tids]
    ys      = [targets[t]["Y"] for t in tids]
    zs      = [targets[t]["Z"] for t in tids]
    tcodes  = [targets[t]["WTATargetInfoCode"] for t in tids]

    type_set = sorted(set(tcodes))
    type_cmap = {tc: plt.cm.tab10(i / max(len(type_set) - 1, 1))
                 for i, tc in enumerate(type_set)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Target Overview", fontsize=12, fontweight="bold")

    # threat score distribution
    ax = axes[0, 0]
    ax.hist(threats, bins=20, color="crimson", edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(threats), color="navy", linestyle="--", linewidth=1,
               label=f"mean={np.mean(threats):.2f}")
    ax.set_title("Threat Score Distribution")
    ax.set_xlabel("Threat score (w_j)")
    ax.set_ylabel("# Targets")
    ax.legend(fontsize=8)

    # speed distribution
    ax = axes[0, 1]
    ax.hist(speeds, bins=20, color="mediumpurple", edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(speeds), color="navy", linestyle="--", linewidth=1,
               label=f"mean={np.mean(speeds):.0f}")
    ax.set_title("Target Speed Distribution")
    ax.set_xlabel("Speed")
    ax.set_ylabel("# Targets")
    ax.legend(fontsize=8)

    # spatial scatter XY, coloured by threat
    ax = axes[1, 0]
    sc_plot = ax.scatter(xs, ys, c=threats, cmap="YlOrRd", s=40,
                         edgecolors="gray", linewidths=0.3)
    plt.colorbar(sc_plot, ax=ax, label="Threat score")
    ax.set_title("Target Positions (X vs Y), colour = threat")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # type breakdown
    ax = axes[1, 1]
    tc_counts = defaultdict(int)
    for tc in tcodes:
        tc_counts[tc] += 1
    codes = sorted(tc_counts)
    vals  = [tc_counts[c] for c in codes]
    colors = [type_cmap[c] for c in codes]
    bars = ax.barh(codes, vals, color=colors)
    ax.set_title("Target Types")
    ax.set_xlabel("# Targets")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=8)

    plt.tight_layout()
    return fig


def fig_network(weapons, targets, weapons_per_target, targets_per_weapon, p_ij):
    wpt = [len(weapons_per_target.get(t, [])) for t in sorted(targets)]
    tpw = [len(targets_per_weapon.get(w, [])) for w in sorted(weapons)]
    p_vals = list(p_ij.values())

    # threat vs weapon_count scatter
    threats = [targets[t]["ThreatScore"] for t in sorted(targets)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Engagement Network", fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(wpt, bins=range(0, max(wpt) + 2), color="teal", edgecolor="white", linewidth=0.4,
            align="left")
    ax.set_title("Weapons per Target Distribution")
    ax.set_xlabel("# Weapons that can engage target")
    ax.set_ylabel("# Targets")
    ax.axvline(np.mean(wpt), color="red", linestyle="--", label=f"mean={np.mean(wpt):.1f}")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(tpw, bins=range(0, max(tpw) + 2), color="darkcyan", edgecolor="white", linewidth=0.4,
            align="left")
    ax.set_title("Targets per Weapon Distribution")
    ax.set_xlabel("# Targets a weapon can engage")
    ax.set_ylabel("# Weapons")
    ax.axvline(np.mean(tpw), color="red", linestyle="--", label=f"mean={np.mean(tpw):.1f}")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(p_vals, bins=20, color="forestgreen", edgecolor="white", linewidth=0.4)
    ax.set_title(f"Kill Probability Distribution (p_ij)\n{len(p_vals)} pairs total")
    ax.set_xlabel("p_kill per burst")
    ax.set_ylabel("# Pairs")

    ax = axes[1, 1]
    ax.scatter(wpt, threats, alpha=0.6, s=30, c="navy", edgecolors="none")
    ax.set_title("Threat Score vs # Engaging Weapons")
    ax.set_xlabel("# Weapons that can engage")
    ax.set_ylabel("Threat score")
    # highlight targets with very few engaging weapons
    for t_idx, t in enumerate(sorted(targets)):
        n = wpt[t_idx]
        if n <= 4:
            ax.annotate(f"T{t}", (n, threats[t_idx]), fontsize=7, color="red",
                        xytext=(4, 4), textcoords="offset points")

    plt.tight_layout()
    return fig


def fig_windows(weapons, windows, p_ij):
    window_widths: List[float] = []
    feasible_slots: List[int]  = []
    slot_x_p: List[Tuple[int, float]] = []   # (slots, p)
    zero_slot_pairs = 0
    total_pairs = len(windows)

    for (wid, tid), (a, b) in windows.items():
        d = weapons[wid]["_burst_dur"]
        width = b - a
        slots = int(width / d) if width >= d else 0
        if slots == 0:
            zero_slot_pairs += 1
        window_widths.append(width)
        feasible_slots.append(slots)
        slot_x_p.append((slots, p_ij[(wid, tid)]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Engagement Window Analysis", fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(window_widths, bins=30, color="mediumpurple", edgecolor="white", linewidth=0.4)
    ax.set_title("Window Width Distribution  (b - a)")
    ax.set_xlabel("Window width [s]")
    ax.set_ylabel("# Pairs")
    ax.axvline(np.mean(window_widths), color="red", linestyle="--",
               label=f"mean={np.mean(window_widths):.1f}s")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    max_slots = max(feasible_slots)
    bins = range(0, min(max_slots, 50) + 2)
    ax.hist([s for s in feasible_slots if s <= 50], bins=bins,
            color="darkorange", edgecolor="white", linewidth=0.4, align="left")
    ax.set_title(f"Feasible Slots per Pair  (0-slots = {zero_slot_pairs}/{total_pairs} = "
                 f"{zero_slot_pairs/total_pairs*100:.1f}% infeasible)")
    ax.set_xlabel("# Schedulable burst slots  floor((b-a)/d)")
    ax.set_ylabel("# Pairs")

    ax = axes[1, 0]
    slots_arr = np.array([s for s, _ in slot_x_p if s <= 50])
    p_arr     = np.array([p for s, p in slot_x_p if s <= 50])
    ax.scatter(slots_arr, p_arr, alpha=0.3, s=8, color="steelblue", edgecolors="none")
    ax.set_title("p_kill vs Feasible Slots\n(each dot = one weapon-target pair)")
    ax.set_xlabel("Feasible slots")
    ax.set_ylabel("p_kill per burst")

    # Burst duration breakdown
    ax = axes[1, 1]
    burst_durs = sorted({weapons[w]["_burst_dur"] for w in weapons})
    labels, frac_zero, avg_slots = [], [], []
    for d in burst_durs:
        wids_with_d = [w for w in weapons if weapons[w]["_burst_dur"] == d]
        pairs_d = [(wid, tid) for (wid, tid) in windows if weapons[wid]["_burst_dur"] == d]
        if not pairs_d:
            continue
        slots_d = [int((windows[pair][1] - windows[pair][0]) / d)
                   if (windows[pair][1] - windows[pair][0]) >= d else 0
                   for pair in pairs_d]
        labels.append(f"d={d:.1f}s\n({len(wids_with_d)}W)")
        frac_zero.append(sum(s == 0 for s in slots_d) / len(slots_d) * 100)
        avg_slots.append(np.mean(slots_d))

    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w/2, frac_zero, w, label="% infeasible pairs", color="salmon")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w/2, avg_slots, w, label="avg feasible slots", color="skyblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("% infeasible pairs", color="salmon")
    ax2.set_ylabel("Avg slots per pair", color="steelblue")
    ax.set_title("By Burst Duration: Infeasibility & Coverage")
    lines = [mpatches.Patch(color="salmon", label="% infeasible"),
             mpatches.Patch(color="skyblue", label="avg slots")]
    ax.legend(handles=lines, fontsize=7)

    plt.tight_layout()
    return fig


def fig_coverage_heatmap(weapons, targets, windows, p_ij):
    """Weapon × Target coverage matrix."""
    wids = sorted(weapons)
    tids = sorted(targets)
    w_idx = {w: i for i, w in enumerate(wids)}
    t_idx = {t: i for i, t in enumerate(tids)}

    # slots matrix
    slots_mat = np.zeros((len(wids), len(tids)))
    pkill_mat = np.zeros((len(wids), len(tids)))
    for (wid, tid), (a, b) in windows.items():
        d = weapons[wid]["_burst_dur"]
        slots = int((b - a) / d) if (b - a) >= d else 0
        slots_mat[w_idx[wid], t_idx[tid]] = slots
        pkill_mat[w_idx[wid], t_idx[tid]] = p_ij[(wid, tid)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Weapon × Target Coverage Heatmaps", fontsize=12, fontweight="bold")

    im1 = axes[0].imshow(slots_mat, aspect="auto", cmap="YlGn", interpolation="none")
    axes[0].set_title("Feasible Slots (green = more)")
    axes[0].set_xlabel("Target ID (index)")
    axes[0].set_ylabel("Weapon ID (index)")
    plt.colorbar(im1, ax=axes[0], shrink=0.7, label="slots")

    im2 = axes[1].imshow(pkill_mat, aspect="auto", cmap="Reds",
                          vmin=0, vmax=1, interpolation="none")
    axes[1].set_title("p_kill per Burst (0 = no engagement window)")
    axes[1].set_xlabel("Target ID (index)")
    axes[1].set_ylabel("Weapon ID (index)")
    plt.colorbar(im2, ax=axes[1], shrink=0.7, label="p_kill")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WTA Scenario EDA")
    parser.add_argument("scenario", nargs="?",
                        default="/workspaces/WTA/data/scenario_001.json")
    parser.add_argument("--out", default="eda.png",
                        help="Output file: .pdf saves all figures; .png saves eda_fig1.png etc.")
    args = parser.parse_args()

    print(f"Loading {args.scenario} ...")
    weapons, targets, windows, p_ij = load(args.scenario)
    weapons_per_target, targets_per_weapon, ww, fs, pv = derive(weapons, targets, windows, p_ij)

    print(f"  {len(weapons)} weapons, {len(targets)} targets, {len(windows)} feasible pairs")
    print(f"  {sum(1 for s in fs if s == 0)} pairs infeasible (window < burst_dur)")

    figs = [
        fig_weapons(weapons),
        fig_targets(targets),
        fig_network(weapons, targets, weapons_per_target, targets_per_weapon, p_ij),
        fig_windows(weapons, windows, p_ij),
        fig_coverage_heatmap(weapons, targets, windows, p_ij),
    ]

    if args.out.endswith(".pdf"):
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(args.out) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        print(f"Saved {args.out}")
    else:
        base, ext = args.out.rsplit(".", 1)
        for i, fig in enumerate(figs, 1):
            path = f"{base}_fig{i}.{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
