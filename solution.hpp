#pragma once
#include "wtv.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Pair key helpers  (weapon_id, target_id) -> flat int for unordered_map
// Using a combined 64-bit key: hi 32 = weapon_id, lo 32 = target_id
// ---------------------------------------------------------------------------
inline uint64_t pair_key(int wid, int tid) {
    return (static_cast<uint64_t>(wid) << 32) | static_cast<uint32_t>(tid);
}

// ---------------------------------------------------------------------------
// Interval type
// ---------------------------------------------------------------------------
struct Interval {
    double s, e;
    bool operator<(const Interval& o) const { return s < o.s; }
};

// ---------------------------------------------------------------------------
// Assignment record (output)
// ---------------------------------------------------------------------------
struct Assignment {
    int    vessel_id;
    int    weapon_id;
    int    target_id;
    int    ammo_used;
    double pkill;
    double fire_time;
    double end_time;
    std::vector<double> fire_times;  // individual burst start times
};

// ---------------------------------------------------------------------------
// Solution
// ---------------------------------------------------------------------------
struct Solution {
    // --- assignment decisions ---
    std::unordered_map<uint64_t, int>                 k;           // (wid,tid) -> burst count
    std::unordered_map<uint64_t, std::vector<double>> fire_times;  // (wid,tid) -> fire time list

    // --- mutable weapon state ---
    std::unordered_map<int, std::vector<Interval>> free;           // wid -> sorted free intervals
    std::unordered_map<int, int>                   remaining_ammo; // wid -> remaining ammo
    std::unordered_map<uint64_t, int>              cap;            // (wid,tid) -> scheduling slots
    std::unordered_map<uint64_t, double>           first_slot_cache; // (wid,tid) -> earliest fire time
                                                                     // NaN = infeasible

    // --- mutable target kill-chain state ---
    std::unordered_map<int, double> survival_rate; // tid -> Π(1-p)^k
    std::unordered_map<int, double> threat_score;  // tid -> w_j (static copy)

    // --- incremental caches (kept consistent by commit/uncommit) ---
    double _obj = 0.0;                        // current objective value
    std::unordered_map<int, int> engaged_count; // tid -> total bursts assigned so far

    // --- static lookups (shared via const pointers; never mutated after init) ---
    const std::unordered_map<uint64_t, double>*                  p_ij         = nullptr;
    const std::unordered_map<uint64_t, std::pair<double,double>>* windows      = nullptr;
    std::unordered_map<int, std::vector<int>>                    weapon_targets; // mutable (pruned)
    const std::unordered_map<int, double>*                       burst_dur     = nullptr;
    const std::unordered_map<int, int>*                          max_shots     = nullptr;
    const std::unordered_map<int, int>*                          vessel_id_map = nullptr;

    // ------------------------------------------------------------------ helpers

    static constexpr double NO_SLOT = std::numeric_limits<double>::quiet_NaN();

    double survival(int tid) const {
        return survival_rate.at(tid);
    }

    double objective() const { return _obj; }

    double first_slot(int wid, int tid) const {
        auto it = first_slot_cache.find(pair_key(wid, tid));
        return it != first_slot_cache.end() ? it->second : NO_SLOT;
    }

    // ------------------------------------------------------------------ _recompute_cap

    void _recompute_cap(int wid) {
        double d          = burst_dur->at(wid);
        auto&  fv         = free.at(wid);
        auto&  tgts       = weapon_targets[wid];
        std::vector<int>  alive;
        alive.reserve(tgts.size());

        for (int tid : tgts) {
            uint64_t key    = pair_key(wid, tid);
            auto [a, b]     = windows->at(key);
            double first_t  = NO_SLOT;
            int    slots    = 0;

            if (!fv.empty() && b - a >= d) {
                // bisect_left for interval with start >= a; step back one
                auto it = std::lower_bound(fv.begin(), fv.end(), Interval{a, 0},
                    [](const Interval& iv, const Interval& val){ return iv.s < val.s; });
                if (it != fv.begin()) --it;

                for (; it != fv.end(); ++it) {
                    if (it->s >= b) break;
                    double lo     = std::max(a, it->s);
                    double hi     = std::min(b, it->e);
                    double length = hi - lo;
                    if (length >= d) {
                        if (std::isnan(first_t)) first_t = lo;
                        slots += static_cast<int>(length / d);
                    }
                }
            }

            if (slots > 0) {
                cap[key]              = slots;
                first_slot_cache[key] = first_t;
                alive.push_back(tid);
            } else {
                cap.erase(key);
                first_slot_cache.erase(key);
            }
        }
        tgts = std::move(alive);
    }

    // ------------------------------------------------------------------ commit

    void commit(int wid, int tid, double t) {
        uint64_t key = pair_key(wid, tid);
        double   d   = burst_dur->at(wid);
        double   end = t + d;

        // single lookup for p_ij (opt #6)
        double p = 0.0;
        { auto pit = p_ij->find(key); if (pit != p_ij->end()) p = pit->second; }

        k[key]++;
        fire_times[key].push_back(t);

        // splice [t, end] out of the covering interval in free[wid]
        auto& fv = free.at(wid);
        // bisect_right: last interval with start <= t
        auto it = std::upper_bound(fv.begin(), fv.end(), Interval{t, std::numeric_limits<double>::infinity()},
            [](const Interval& val, const Interval& iv){ return val.s < iv.s; });
        if (it != fv.begin()) --it;  // covering interval

        double fs = it->s, fe = it->e;
        it = fv.erase(it);  // remove covering interval
        if (end < fe) it = fv.insert(it, {end, fe});
        if (fs < t)   it = fv.insert(it, {fs, t});

        remaining_ammo[wid]--;

        // update _obj and survival incrementally (opt #1)
        double w = threat_score.at(tid);
        _obj -= w * survival_rate.at(tid);
        survival_rate[tid] *= (1.0 - p);
        _obj += w * survival_rate.at(tid);

        // update engaged_count (opt #5)
        engaged_count[tid]++;

        _recompute_cap(wid);
    }

    // ------------------------------------------------------------------ uncommit

    // Reverse the effect of commit(wid, tid, t).
    // Precondition: (wid, tid, t) was previously committed.
    void uncommit(int wid, int tid, double t) {
        uint64_t key = pair_key(wid, tid);
        double   d   = burst_dur->at(wid);
        double   end = t + d;

        // single lookup for p_ij (opt #6)
        double p = 0.0;
        { auto pit = p_ij->find(key); if (pit != p_ij->end()) p = pit->second; }

        // 1. Restore burst count
        auto k_it = k.find(key);
        if (k_it == k.end() || k_it->second <= 0)
            throw std::runtime_error("uncommit: missing burst count for (wid,tid)");
        if (--k_it->second == 0) k.erase(k_it);

        // 2. Remove fire time t
        {
            auto ft_it = fire_times.find(key);
            if (ft_it == fire_times.end())
                throw std::runtime_error("uncommit: missing fire_times entry for (wid,tid)");
            auto& ft = ft_it->second;
            auto  it = std::find(ft.begin(), ft.end(), t);
            if (it == ft.end())
                throw std::runtime_error("uncommit: fire time not found for (wid,tid,t)");
            ft.erase(it);
            if (ft.empty()) fire_times.erase(key);
        }

        // 3. Restore ammo, survival, _obj, and engaged_count
        remaining_ammo[wid]++;

        // update _obj and survival incrementally (opt #1)
        double w = threat_score.at(tid);
        _obj -= w * survival_rate.at(tid);
        if (p < 1.0) survival_rate[tid] /= (1.0 - p);
        _obj += w * survival_rate.at(tid);

        // update engaged_count (opt #5)
        auto ec_it = engaged_count.find(tid);
        if (ec_it != engaged_count.end()) {
            if (--ec_it->second <= 0) engaged_count.erase(ec_it);
        }

        // 4. Re-merge [t, end] back into free[wid].
        //    Use a generic insert+coalesce instead of assuming exact left/right
        //    fragments are still present. This is robust under repeated local edits.
        auto& fv = free.at(wid);
        constexpr double eps = 1e-9;
        fv.push_back({t, end});
        std::sort(fv.begin(), fv.end());

        std::vector<Interval> merged;
        merged.reserve(fv.size());
        for (const auto& iv : fv) {
            if (merged.empty()) {
                merged.push_back(iv);
                continue;
            }
            auto& last = merged.back();
            if (iv.s <= last.e + eps) {
                last.e = std::max(last.e, iv.e);
            } else {
                merged.push_back(iv);
            }
        }
        fv.swap(merged);

        // 5. Restore tid in weapon_targets[wid] if it was pruned, then recompute cap.
        auto& tgts = weapon_targets[wid];
        if (std::find(tgts.begin(), tgts.end(), tid) == tgts.end())
            tgts.push_back(tid);
        _recompute_cap(wid);
    }

    // ------------------------------------------------------------------ copy

    Solution copy() const {
        Solution s;
        s.k              = k;
        s.fire_times     = fire_times;
        s.free           = free;
        s.remaining_ammo = remaining_ammo;
        s.cap            = cap;
        s.first_slot_cache = first_slot_cache;
        s.survival_rate  = survival_rate;
        s.threat_score   = threat_score;
        s.weapon_targets = weapon_targets;
        s._obj           = _obj;
        s.engaged_count  = engaged_count;
        // shared const pointers
        s.p_ij         = p_ij;
        s.windows      = windows;
        s.burst_dur    = burst_dur;
        s.max_shots    = max_shots;
        s.vessel_id_map = vessel_id_map;
        return s;
    }

    // ------------------------------------------------------------------ empty (factory)

    static Solution empty(
        const std::vector<Weapon>&  weapons,
        const std::vector<Target>&  targets,
        const std::unordered_map<uint64_t, double>&                   p_ij_in,
        const std::unordered_map<uint64_t, std::pair<double,double>>& windows_in,
        const std::unordered_map<int, double>&  burst_dur_in,
        const std::unordered_map<int, int>&     max_shots_in,
        const std::unordered_map<int, int>&     vessel_id_in,
        double horizon)
    {
        Solution sol;
        sol.p_ij          = &p_ij_in;
        sol.windows       = &windows_in;
        sol.burst_dur     = &burst_dur_in;
        sol.max_shots     = &max_shots_in;
        sol.vessel_id_map = &vessel_id_in;

        for (const auto& w : weapons) {
            sol.remaining_ammo[w.id] = w.ammo;
            sol.free[w.id]           = {{0.0, horizon}};
        }
        for (const auto& t : targets) {
            sol.survival_rate[t.id] = 1.0;
            sol.threat_score[t.id]  = t.threat_score;
            sol._obj += t.threat_score;  // survival starts at 1.0
        }
        for (auto& [key, ab] : windows_in) {
            int    wid = static_cast<int>(key >> 32);
            int    tid = static_cast<int>(key & 0xFFFFFFFF);
            double a   = ab.first, b = ab.second;
            double d   = burst_dur_in.at(wid);
            if (b - a < d) continue;  // prune statically infeasible
            sol.weapon_targets[wid].push_back(tid);
            sol.cap[key]               = static_cast<int>((b - a) / d);
            sol.first_slot_cache[key]  = a;
        }
        return sol;
    }

    // ------------------------------------------------------------------ assignments (output)

    std::vector<Assignment> assignments() const {
        std::vector<Assignment> result;
        for (auto& [key, count] : k) {
            if (count == 0) continue;
            int    wid  = static_cast<int>(key >> 32);
            int    tid  = static_cast<int>(key & 0xFFFFFFFF);
            double p    = 0.0;
            { auto pit = p_ij->find(key); if (pit != p_ij->end()) p = pit->second; }
            double d    = burst_dur->at(wid);
            const auto& times = fire_times.at(key);
            double tmin = *std::min_element(times.begin(), times.end());
            double tmax = *std::max_element(times.begin(), times.end());
            result.push_back({
                vessel_id_map->at(wid), wid, tid, count,
                1.0 - std::pow(1.0 - p, count),
                tmin, tmax + d,
                times  // individual fire times
            });
        }
        return result;
    }
};
