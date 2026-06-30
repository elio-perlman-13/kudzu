#pragma once
#include "solution.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>


static inline int slots_overlap(double a, double b, double s, double e, double d) {
    double lo     = std::max(a, s);
    double hi     = std::min(b, e);
    double length = hi - lo;
    return length >= d ? static_cast<int>(length / d) : 0;
}

static double score(const Solution& sol, int wid, int tid, double t,
                    int exclusive_cnt) {
    (void)t;
    (void)wid;
    (void)exclusive_cnt;
    return sol.survival(tid) * sol.threat_score.at(tid);
}

// grasp_construction — incremental-scoring GRASP construction 

Solution& grasp_construction(Solution& sol, double alpha, std::mt19937& rng) {
    std::unordered_map<int, std::vector<int>> target_weapons;
    for (auto& [wid, tgts] : sol.weapon_targets)
        for (int tid : tgts)
            target_weapons[tid].push_back(wid);

    struct CS { double t, sc; };
    std::unordered_map<int, std::unordered_map<int, CS>> cache;

    auto try_score = [&](int wid, int tid) {
        uint64_t key    = pair_key(wid, tid);
        auto     cap_it = sol.cap.find(key);
        if (cap_it == sol.cap.end()) return;
        int ammo = sol.remaining_ammo.at(wid);
        if (ammo <= 0) return;
        int k_val = 0;
        if (auto ki = sol.k.find(key); ki != sol.k.end()) k_val = ki->second;
        if (k_val >= sol.max_shots->at(wid)) return;
        double t = sol.first_slot(wid, tid);
        if (std::isnan(t)) return;

        int excl = 0;
        auto wt_it = sol.weapon_targets.find(wid);
        if (wt_it != sol.weapon_targets.end()) {
            for (int j : wt_it->second) {
                auto tw_it = target_weapons.find(j);
                if (tw_it != target_weapons.end() && tw_it->second.size() == 1)
                    ++excl;
            }
        }

        cache[wid][tid] = {t, score(sol, wid, tid, t, excl)};
    };

    for (auto& [wid, tgts] : sol.weapon_targets)
        for (int tid : tgts)
            try_score(wid, tid);

    while (!cache.empty()) {

        double best_sc = -std::numeric_limits<double>::infinity();
        for (auto& [wid, inner] : cache)
            for (auto& [tid, cs] : inner)
                best_sc = std::max(best_sc, cs.sc);
        if (best_sc <= 0.0) break;

        constexpr double eps = 1e-12;

        // Stage 1: build target RCL and sample 
        std::unordered_map<int, double> target_best;
        for (auto& [wid, inner] : cache)
            for (auto& [tid, cs] : inner)
                target_best[tid] = std::max(target_best[tid], cs.sc);

        double target_threshold = alpha * best_sc;
        std::vector<int> target_rcl;
        target_rcl.reserve(target_best.size());
        for (auto& [tid, sc] : target_best)
            if (sc >= target_threshold - eps)
                target_rcl.push_back(tid);

        if (target_rcl.empty()) break;

        int selected_tid = target_rcl[
            std::uniform_int_distribution<int>(0, static_cast<int>(target_rcl.size()) - 1)(rng)];

        //  Stage 2: pick best weapon for selected_tid (deterministic)
        bool   have_choice  = false;
        int    chosen_wid   = -1;
        int    chosen_tid   = selected_tid;
        double chosen_t     = 0.0;
        int    best_scarcity = std::numeric_limits<int>::max();
        double best_p        = -1.0;

        for (auto& [wid, inner] : cache) {
            auto it = inner.find(selected_tid);
            if (it == inner.end()) continue;
            const CS& cs = it->second;

            int    scarcity = static_cast<int>(inner.size());
            double p_this   = 0.0;
            auto   pit      = sol.p_ij->find(pair_key(wid, selected_tid));
            if (pit != sol.p_ij->end()) p_this = pit->second;

            if (!have_choice
                || scarcity < best_scarcity
                || (scarcity == best_scarcity && p_this > best_p + eps)
                || (scarcity == best_scarcity && std::fabs(p_this - best_p) <= eps
                    && (cs.t < chosen_t - eps
                        || (std::fabs(cs.t - chosen_t) <= eps && wid < chosen_wid)))) {
                have_choice   = true;
                chosen_wid    = wid;
                chosen_t      = cs.t;
                best_scarcity = scarcity;
                best_p        = p_this;
            }
        }

        if (!have_choice) break;

        std::vector<int> dirty = {chosen_wid};
        if (auto it = target_weapons.find(chosen_tid); it != target_weapons.end())
            for (int w : it->second)
                if (w != chosen_wid) dirty.push_back(w);

        for (int w : dirty) cache.erase(w);

        sol.commit(chosen_wid, chosen_tid, chosen_t);

        for (int w : dirty) {
            auto tgts_it = sol.weapon_targets.find(w);
            if (tgts_it == sol.weapon_targets.end()) continue;
            for (int j : tgts_it->second)
                try_score(w, j);
            if (cache.count(w) && cache[w].empty()) cache.erase(w);
        }
    }
    return sol;
}

Solution grasp(
    const std::vector<Weapon>&  weapons,
    const std::vector<Target>&  targets,
    const std::unordered_map<uint64_t, double>&                   p_ij,
    const std::unordered_map<uint64_t, std::pair<double,double>>& windows,
    const std::unordered_map<int, double>&  burst_dur,
    const std::unordered_map<int, int>&     max_shots,
    const std::unordered_map<int, int>&     vessel_id_map,
    double horizon,
    double alpha    = 0.15,
    int    restarts = 1,
    uint32_t seed   = 42)
{
    std::mt19937 rng(seed);
    Solution     best;
    bool         have_best = false;

    for (int r = 0; r < restarts; ++r) {
        Solution sol = Solution::empty(
            weapons, targets, p_ij, windows,
            burst_dur, max_shots, vessel_id_map, horizon);
        grasp_construction(sol, alpha, rng);
        if (!have_best || lex_better(lex_score(sol), lex_score(best))) {
            best      = sol;
            have_best = true;
        }
    }
    assert(have_best);
    return best;
}