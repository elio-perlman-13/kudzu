#!/usr/bin/env python3
"""
check_solution.py — Validate a WTA solution JSON against its scenario.

Usage:
    python check_solution.py [scenario.json] [solution.json]

Checks:
  1. Every (weapon, target) pair has a valid engagement window.
  2. Per-weapon fire times are non-overlapping (timeline feasibility).
  3. Per-weapon ammo limits respected.
  4. Per-(weapon, target) burst cap M_i respected.
  5. Each fire time falls within the engagement window [a, b-d].
  6. Reported PKill matches  1 - (1-p)^ammo_used.
  7. Reported EndTime matches last_fire_time + burst_duration.
  8. Reported objective matches computed Σ_j w_j Π_i (1-p_ij)^k_ij.
  9. All weapon/target IDs exist in the scenario.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import isclose
from typing import Dict, List, Tuple


def load_scenario(path: str):
    with open(path) as f:
        data = json.load(f)

    weapon_info_by_code = {d["Code"]: d for d in data["weapon_infos"]}

    prob: Dict[Tuple[str, str], float] = {}
    for row in data["probability_table"]:
        prob[(row["WTAWeaponInfoCode"], row["WTATargetInfoCode"])] = row["Score"]

    req = data["assignment_request"]

    weapons = {}
    for d in req["weapons"]:
        d = dict(d)
        wi = weapon_info_by_code[d["WTAWeaponInfoCode"]]
        d["_burst_dur"] = wi["BurstInterval"] + wi["ReloadTime"]
        d["_max_shots"] = wi["MaxShotsPerTarget"]
        weapons[d["ID"]] = d

    targets = {d["ID"]: d for d in req["targets"]}

    windows: Dict[Tuple[int, int], Tuple[float, float]] = {}
    p_ij:    Dict[Tuple[int, int], float] = {}
    for key_str, (a, b) in data["engagement_windows"].items():
        wid, tid = map(int, key_str.split("_"))
        if wid not in weapons or tid not in targets:
            continue
        wcode = weapons[wid]["WTAWeaponInfoCode"]
        tcode = targets[tid]["WTATargetInfoCode"]
        p = prob.get((wcode, tcode), 0.0)
        if p <= 0.0:
            continue
        windows[(wid, tid)] = (a, b)
        p_ij[(wid, tid)] = p

    return weapons, targets, windows, p_ij


def check(scenario_path: str, solution_path: str) -> bool:
    weapons, targets, windows, p_ij = load_scenario(scenario_path)

    with open(solution_path) as f:
        sol = json.load(f)

    assignments: List[dict] = sol.get("assignments", [])
    reported_obj = sol.get("objective")

    errors:   List[str] = []
    warnings: List[str] = []

    by_weapon_target: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    by_weapon:        Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    for idx, a in enumerate(assignments):
        wid       = a["WTAWeaponID"]
        tid       = a["WTATargetID"]
        fire      = a["FireTime"]
        end       = a["EndTime"]
        ammo      = a["AmmoUsed"]
        pkill_rep = a["PKill"]

        # 1. IDs exist
        if wid not in weapons:
            errors.append(f"[#{idx}] unknown WTAWeaponID={wid}")
            continue
        if tid not in targets:
            errors.append(f"[#{idx}] unknown WTATargetID={tid}")
            continue

        d = weapons[wid]["_burst_dur"]

        # 2. Engagement window exists
        key = (wid, tid)
        if key not in windows:
            errors.append(f"[#{idx}] (weapon={wid}, target={tid}) not in engagement windows / p=0")
            continue

        a_win, b_win = windows[key]
        p = p_ij[key]

        # Individual burst fire times
        individual_times: List[float] = a.get("FireTimes") or [fire]
        last_fire = max(individual_times)

        # 3. Each burst within window
        for ft in individual_times:
            if ft < a_win - 1e-6:
                errors.append(
                    f"[#{idx}] weapon={wid} target={tid}: burst at {ft:.4f} < window_start={a_win:.4f}"
                )
            if ft + d > b_win + 1e-6:
                errors.append(
                    f"[#{idx}] weapon={wid} target={tid}: burst ends {ft+d:.4f} > window_end={b_win:.4f}"
                )

        # 4. EndTime = last_fire + d
        expected_end = last_fire + d
        if not isclose(end, expected_end, rel_tol=1e-4, abs_tol=1e-4):
            warnings.append(
                f"[#{idx}] weapon={wid} target={tid}: EndTime={end:.4f} expected {expected_end:.4f}"
            )

        # 5. AmmoUsed matches FireTimes count
        if len(individual_times) != ammo:
            warnings.append(
                f"[#{idx}] weapon={wid} target={tid}: AmmoUsed={ammo} but {len(individual_times)} FireTimes"
            )

        # 6. PKill consistency
        expected_pkill = 1.0 - (1.0 - p) ** ammo
        if not isclose(pkill_rep, expected_pkill, rel_tol=1e-4, abs_tol=1e-4):
            warnings.append(
                f"[#{idx}] weapon={wid} target={tid}: PKill={pkill_rep:.6f} expected {expected_pkill:.6f}"
            )

        by_weapon_target[key].append(a)
        for ft in individual_times:
            by_weapon[wid].append((ft, ft + d))

    # 7. Per-(weapon, target) burst cap M
    for (wid, tid), recs in by_weapon_target.items():
        total = sum(r["AmmoUsed"] for r in recs)
        M = weapons[wid]["_max_shots"]
        if total > M:
            errors.append(f"weapon={wid} target={tid}: total bursts={total} > M={M}")

    # 8. Per-weapon ammo limit
    ammo_used: Dict[int, int] = defaultdict(int)
    for (wid, _), recs in by_weapon_target.items():
        ammo_used[wid] += sum(r["AmmoUsed"] for r in recs)
    for wid, used in ammo_used.items():
        cap = weapons[wid]["Ammo"]
        if used > cap:
            errors.append(f"weapon={wid}: total bursts={used} > ammo={cap}")

    # 9. Timeline non-overlap per weapon
    EPS = 1e-6
    for wid, intervals in by_weapon.items():
        sorted_iv = sorted(intervals)
        for i in range(len(sorted_iv) - 1):
            _, e1 = sorted_iv[i]
            s2, _ = sorted_iv[i + 1]
            if s2 < e1 - EPS:
                errors.append(
                    f"weapon={wid}: overlapping bursts [{sorted_iv[i][0]:.3f},{e1:.3f}) "
                    f"and [{s2:.3f},{sorted_iv[i+1][1]:.3f})"
                )

    # 10. Recompute objective
    k: Dict[Tuple[int, int], int] = defaultdict(int)
    for (wid, tid), recs in by_weapon_target.items():
        k[(wid, tid)] = sum(r["AmmoUsed"] for r in recs)

    survival: Dict[int, float] = {tid: 1.0 for tid in targets}
    for (wid, tid), bursts in k.items():
        survival[tid] *= (1.0 - p_ij.get((wid, tid), 0.0)) ** bursts

    computed_obj = sum(targets[tid]["ThreatScore"] * survival[tid] for tid in targets)

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    feasible = len(errors) == 0

    print(f"Scenario   : {scenario_path}")
    print(f"Solution   : {solution_path}")
    print(f"Assignments: {len(assignments)}")
    print(f"Computed objective : {computed_obj:.6f}")
    if reported_obj is not None:
        match = "OK" if isclose(computed_obj, reported_obj, rel_tol=1e-5, abs_tol=1e-5) else "MISMATCH"
        print(f"Reported objective : {reported_obj:.6f}  [{match}]")
    print(f"Feasible   : {'YES' if feasible else 'NO'}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  [ERROR] {e}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN]  {w}")

    return feasible


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WTA solution feasibility checker")
    parser.add_argument("scenario", nargs="?",
                        default="/workspaces/WTA/data/scenario_001.json")
    parser.add_argument("solution", nargs="?",
                        default="/workspaces/WTA/data/scenario_001_solution.json")
    args = parser.parse_args()
    sys.exit(0 if check(args.scenario, args.solution) else 1)
