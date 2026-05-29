#pragma once

#include <cstdint>

#include "portfolio.hpp"
#include "solution.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace perturbation {

// Ruin strategy used by macro LLHs.
// *_BURST variants select individual bursts (fine-grained).
// *_PAIR  variants select (wid,tid) pairs and remove ALL their bursts (coarser, more coherent).
enum class RuinRule {
	RANDOM,          // random burst selection
	OVER_COVERED,    // burst-level: targets with most shots
	CONGESTED,       // burst-level: weapons with least free time
	LOW_MARGINAL,    // burst-level: lowest marginal contribution
	RANDOM_PAIR,     // random pair; remove all bursts on that (wid,tid)
	LOW_MARGINAL_PAIR, // weakest pair by total marginal; remove all bursts on it
};

// 14 LLHs total:
// - 12 macro LLHs (M2,M4,M6-M8,M10-M16; M1/M3/M5/M9 removed)
// - 2 local LLHs
enum class LLHId {
	M2_RUIN_RANDOM__H_SURV_THREAT_TIE,
	M4_RUIN_OVER_COVERED__H_EXCLUSIVE_RESERVE,
	M6_RUIN_CONGESTED__H_WINDOW_CLOSURE,
	M7_RUIN_LOW_MARGINAL__H_COVER_FIRST,
	M8_RUIN_CONGESTED__H_BACKLOG_RELIEF,
	M10_RUIN_RANDOM__H_SPREAD_THEN_FOCUS,
	M11_RUIN_CONGESTED__H_ANTI_BOTTLENECK,
	M12_RUIN_OVER_COVERED__H_OPPORTUNITY_LOCK,
	M13_RUIN_LOW_MARGINAL__H_MARGINAL_OBJECTIVE_DROP,
	M14_RUIN_OVER_COVERED__H_KILL_CHAIN_FINISHER,
	M15_RUIN_CONGESTED__H_FUTURE_FLEX_PRESERVER,
	M16_RUIN_RANDOM__H_BASELINE_TWO_STAGE,

	L1_REASSIGN_BEST_DELTA,
	L2_SWAP_BEST_PAIR,
};

inline constexpr int kLLHCount = 14;

struct MacroSpec {
	RuinRule               ruin;
	portfolio::HeuristicId arm;
};

struct ApplyParams {
	// val in [0.2, 1.0]: controls ruin fraction — maps to [5%, 40%] of assignments/pairs.
	double val = 0.2;
};

namespace detail {

struct BurstRef {
	int wid = -1;
	int tid = -1;
	double t = std::numeric_limits<double>::quiet_NaN();
};

static inline double clamped_val(double val) {
	return std::clamp(val, 0.2, 1.0);
}

// Maps val in [0.2,1.0] to a ruin fraction in [5%,40%].
// n is the population size (burst count or pair count).
static inline int ruin_count_from_val(double val, int n) {
	if (n <= 0) return 0;
	double v = clamped_val(val);
	double frac = 0.05 + (v - 0.2) / 0.8 * 0.35;  // [5%, 40%]
	return std::max(1, std::min(n, static_cast<int>(std::round(frac * n))));
}

static inline std::vector<BurstRef> collect_bursts(const Solution& sol) {
	std::vector<BurstRef> out;
	out.reserve(sol.k.size() * 2);
	for (const auto& [key, times] : sol.fire_times) {
		int wid = static_cast<int>(key >> 32);
		int tid = static_cast<int>(key & 0xFFFFFFFF);
		for (double t : times) out.push_back({wid, tid, t});
	}
	return out;
}

static inline double pair_p(const Solution& sol, int wid, int tid) {
	auto it = sol.p_ij->find(pair_key(wid, tid));
	return (it == sol.p_ij->end()) ? 0.0 : it->second;
}

static inline double marginal_uncommit_increase(const Solution& sol, int wid, int tid) {
	// Current objective increase if one burst on (wid, tid) is removed.
	// J = threat * survival; uncommit scales survival by 1/(1-p).
	double p = pair_p(sol, wid, tid);
	if (p <= 0.0) return 0.0;
	if (p >= 1.0) return std::numeric_limits<double>::infinity();
	double s = sol.survival(tid);
	double w = sol.threat_score.at(tid);
	return w * s * (p / (1.0 - p));
}

static inline std::unordered_map<int, int> shots_per_target(const Solution& sol) {
	std::unordered_map<int, int> cnt;
	for (const auto& [key, shots] : sol.k) {
		if (shots <= 0) continue;
		int tid = static_cast<int>(key & 0xFFFFFFFF);
		cnt[tid] += shots;
	}
	return cnt;
}

static inline std::unordered_map<int, double> free_time_per_weapon(const Solution& sol) {
	std::unordered_map<int, double> ft;
	for (const auto& [wid, intervals] : sol.free) {
		double sum = 0.0;
		for (const auto& iv : intervals) sum += std::max(0.0, iv.e - iv.s);
		ft[wid] = sum;
	}
	return ft;
}

static inline bool is_feasible_commit(const Solution& sol, int wid, int tid, double t) {
	if (std::isnan(t)) return false;
	uint64_t key = pair_key(wid, tid);
	if (!sol.cap.count(key)) return false;
	auto ammo_it = sol.remaining_ammo.find(wid);
	if (ammo_it == sol.remaining_ammo.end() || ammo_it->second <= 0) return false;
	int used = 0;
	auto k_it = sol.k.find(key);
	if (k_it != sol.k.end()) used = k_it->second;
	if (used >= sol.max_shots->at(wid)) return false;
	return true;
}

static inline std::vector<portfolio::Candidate> build_candidates(
	const Solution& sol,
	const std::unordered_set<uint64_t>* forbidden = nullptr)
{
	std::vector<portfolio::Candidate> cands;
	cands.reserve(sol.cap.size());
	for (const auto& [key, slots] : sol.cap) {
		if (slots <= 0) continue;
		if (forbidden && forbidden->count(key)) continue;
		int wid = static_cast<int>(key >> 32);
		int tid = static_cast<int>(key & 0xFFFFFFFF);
		double t = sol.first_slot(wid, tid);
		if (!is_feasible_commit(sol, wid, tid, t)) continue;
		cands.push_back({wid, tid, t});
	}
	return cands;
}

static inline bool apply_one_arm_commit(
	portfolio::HeuristicId arm,
	Solution& sol,
	std::unordered_map<int, int>* prev_feasible)
{
	auto cands = build_candidates(sol);
	if (cands.empty()) return false;

	int rem_ammo = 0;
	for (const auto& [wid, rem] : sol.remaining_ammo) {
		(void)wid;
		rem_ammo += rem;
	}

	portfolio::SelectContext ctx;
	ctx.phase = (rem_ammo > 0) ? 0.0 : 1.0;  // 0=exploration when ammo free, 1=exploitation when full
	ctx.prev_target_feasible_weapons = prev_feasible && !prev_feasible->empty() ? prev_feasible : nullptr;

	portfolio::SelectResult sel = portfolio::select_candidate(arm, sol, cands, ctx);
	if (!sel.found) {
		sel = portfolio::select_candidate(portfolio::HeuristicId::H_SURV, sol, cands, ctx);
		if (!sel.found) return false;
	}

	if (!is_feasible_commit(sol, sel.cand.wid, sel.cand.tid, sel.cand.t)) return false;
	sol.commit(sel.cand.wid, sel.cand.tid, sel.cand.t);

	if (prev_feasible) {
		prev_feasible->clear();
		for (const auto& c : cands) (*prev_feasible)[c.tid]++;
	}
	return true;
}

// Incremental variant: takes the candidate list by reference and patches only
// the dirty weapon's entries after each commit instead of rebuilding from sol.cap.
// After commit(wid, *): only cap and first_slot_cache entries for wid change
// (via _recompute_cap). All other weapons' candidates remain valid.
static inline bool apply_one_arm_commit_inc(
	portfolio::HeuristicId arm,
	Solution& sol,
	std::vector<portfolio::Candidate>& cands,
	int& rem_ammo,
	std::unordered_map<int, int>* prev_feasible)
{
	if (cands.empty()) return false;

	portfolio::SelectContext ctx;
	ctx.phase = (rem_ammo > 0) ? 0.0 : 1.0;
	ctx.prev_target_feasible_weapons = prev_feasible && !prev_feasible->empty() ? prev_feasible : nullptr;

	portfolio::SelectResult sel = portfolio::select_candidate(arm, sol, cands, ctx);
	if (!sel.found) {
		sel = portfolio::select_candidate(portfolio::HeuristicId::H_SURV, sol, cands, ctx);
		if (!sel.found) return false;
	}

	if (!is_feasible_commit(sol, sel.cand.wid, sel.cand.tid, sel.cand.t)) return false;

	const int dirty_wid = sel.cand.wid;
	sol.commit(dirty_wid, sel.cand.tid, sel.cand.t);
	--rem_ammo;

	// Remove all candidates for dirty_wid (their cap/first_slot changed via _recompute_cap).
	cands.erase(
		std::remove_if(cands.begin(), cands.end(),
			[dirty_wid](const portfolio::Candidate& c) { return c.wid == dirty_wid; }),
		cands.end());

	// Re-add valid candidates for dirty_wid using updated weapon_targets.
	auto wt_it = sol.weapon_targets.find(dirty_wid);
	if (wt_it != sol.weapon_targets.end()) {
		for (int tid : wt_it->second) {
			double t = sol.first_slot(dirty_wid, tid);
			if (!is_feasible_commit(sol, dirty_wid, tid, t)) continue;
			cands.push_back({dirty_wid, tid, t});
		}
	}

	if (prev_feasible) {
		prev_feasible->clear();
		for (const auto& c : cands) (*prev_feasible)[c.tid]++;
	}
	return true;
}

static inline bool reassign_single_burst_best(
	Solution& sol,
	const BurstRef& b,
	bool require_tid_change)
{
	Solution base = sol.copy();
	base.uncommit(b.wid, b.tid, b.t);

	auto cands = build_candidates(base);
	if (cands.empty()) return false;

	bool have = false;
	double best_obj = std::numeric_limits<double>::infinity();
	Solution best_sol;

	for (const auto& c : cands) {
		if (require_tid_change && c.tid == b.tid) continue;
		if (c.wid == b.wid && c.tid == b.tid && std::fabs(c.t - b.t) < 1e-9) continue;

		Solution cand = base.copy();
		if (!is_feasible_commit(cand, c.wid, c.tid, c.t)) continue;
		cand.commit(c.wid, c.tid, c.t);

		double obj = cand.objective();
		if (!have || obj < best_obj) {
			have = true;
			best_obj = obj;
			best_sol = std::move(cand);
		}
	}

	if (!have) return false;
	sol = std::move(best_sol);
	return true;
}

static inline bool commit_best_candidate(
	Solution& sol,
	int forbid_tid)
{
	auto cands = build_candidates(sol);
	if (cands.empty()) return false;

	bool have = false;
	double best_obj = std::numeric_limits<double>::infinity();
	portfolio::Candidate best_c;

	for (const auto& c : cands) {
		if (forbid_tid >= 0 && c.tid == forbid_tid) continue;
		if (!is_feasible_commit(sol, c.wid, c.tid, c.t)) continue;

		Solution cand = sol.copy();
		cand.commit(c.wid, c.tid, c.t);
		double obj = cand.objective();
		if (!have || obj < best_obj) {
			have = true;
			best_obj = obj;
			best_c = c;
		}
	}

	if (!have) return false;
	sol.commit(best_c.wid, best_c.tid, best_c.t);
	return true;
}

} // namespace detail

inline bool apply_macro_move(
	const MacroSpec& spec,
	Solution& sol,
	std::mt19937& rng,
	const ApplyParams& params)
{
	auto bursts = detail::collect_bursts(sol);
	if (bursts.empty()) return false;

	if (spec.ruin == RuinRule::RANDOM_PAIR || spec.ruin == RuinRule::LOW_MARGINAL_PAIR) {
		// --- Pair-level ruin: collect active (wid,tid) pairs, select n_pairs, remove ALL their bursts ---
		// Score each pair by sum of marginal contributions of its bursts.
		std::unordered_map<uint64_t, double> pair_score;
		for (const auto& b : bursts)
			pair_score[pair_key(b.wid, b.tid)] += detail::marginal_uncommit_increase(sol, b.wid, b.tid);

		std::vector<uint64_t> pairs;
		pairs.reserve(pair_score.size());
		for (const auto& [k, _] : pair_score) pairs.push_back(k);

		int n_pairs = detail::ruin_count_from_val(params.val, static_cast<int>(pairs.size()));

		if (spec.ruin == RuinRule::RANDOM_PAIR) {
			std::shuffle(pairs.begin(), pairs.end(), rng);
		} else { // LOW_MARGINAL_PAIR: weakest pairs first
			std::sort(pairs.begin(), pairs.end(), [&](uint64_t a, uint64_t b) {
				return pair_score.at(a) < pair_score.at(b);
			});
		}

		// Uncommit all bursts belonging to the selected pairs.
		std::unordered_set<uint64_t> ruined(pairs.begin(), pairs.begin() + n_pairs);
		for (const auto& b : bursts)
			if (ruined.count(pair_key(b.wid, b.tid)))
				sol.uncommit(b.wid, b.tid, b.t);

	} else {
		// --- Burst-level ruin ---
		int q = detail::ruin_count_from_val(params.val, static_cast<int>(bursts.size()));

		const auto target_shots = detail::shots_per_target(sol);
		const auto free_time    = detail::free_time_per_weapon(sol);

		if (spec.ruin == RuinRule::RANDOM) {
			std::shuffle(bursts.begin(), bursts.end(), rng);
		} else if (spec.ruin == RuinRule::LOW_MARGINAL) {
			std::sort(bursts.begin(), bursts.end(), [&](const detail::BurstRef& a, const detail::BurstRef& b) {
				double da = detail::marginal_uncommit_increase(sol, a.wid, a.tid);
				double db = detail::marginal_uncommit_increase(sol, b.wid, b.tid);
				if (da != db) return da < db;
				return a.wid < b.wid;
			});
		} else if (spec.ruin == RuinRule::OVER_COVERED) {
			std::sort(bursts.begin(), bursts.end(), [&](const detail::BurstRef& a, const detail::BurstRef& b) {
				int sa = target_shots.count(a.tid) ? target_shots.at(a.tid) : 0;
				int sb = target_shots.count(b.tid) ? target_shots.at(b.tid) : 0;
				if (sa != sb) return sa > sb;
				double da = detail::marginal_uncommit_increase(sol, a.wid, a.tid);
				double db = detail::marginal_uncommit_increase(sol, b.wid, b.tid);
				return da < db;
			});
		} else { // CONGESTED
			std::sort(bursts.begin(), bursts.end(), [&](const detail::BurstRef& a, const detail::BurstRef& b) {
				double fa = free_time.count(a.wid) ? free_time.at(a.wid) : 0.0;
				double fb = free_time.count(b.wid) ? free_time.at(b.wid) : 0.0;
				if (fa != fb) return fa < fb;
				double da = detail::marginal_uncommit_increase(sol, a.wid, a.tid);
				double db = detail::marginal_uncommit_increase(sol, b.wid, b.tid);
				return da < db;
			});
		}

		for (int i = 0; i < q; ++i)
			sol.uncommit(bursts[i].wid, bursts[i].tid, bursts[i].t);
	}

	// Rebuild: build candidate list once, then update incrementally after each commit.
	auto cands = detail::build_candidates(sol);
	int rem_ammo = 0;
	for (const auto& [w, r] : sol.remaining_ammo) rem_ammo += r;
	std::unordered_map<int, int> prev_feasible;
	int guard = 0;
	while (guard < 100000) {
		if (!detail::apply_one_arm_commit_inc(spec.arm, sol, cands, rem_ammo, &prev_feasible)) break;
		++guard;
	}

	return true;
}

// L1: best-improvement reassignment — scans ALL active (wid,tid) pairs and ALL alternative
// targets analytically to find the (wid, tid→tid') move with the greatest objective drop.
// Uses analytical delta (no sol.copy during scan); verifies top candidates with actual copy.
inline bool apply_reassign_pair_best(
	Solution& sol,
	std::mt19937& /*rng*/,
	const ApplyParams& /*params*/)
{
	double current_obj = sol.objective();

	struct Move { int wid, tid, tid2, k; double delta; };
	std::vector<Move> candidates;

	for (const auto& [key, times] : sol.fire_times) {
		if (times.empty()) continue;
		int wid = static_cast<int>(key >> 32);
		int tid = static_cast<int>(key  & 0xFFFFFFFF);
		int k   = static_cast<int>(times.size());

		double p = detail::pair_p(sol, wid, tid);
		if (p <= 0.0 || p >= 1.0) continue;
		double surv_tid   = sol.survival(tid);
		double threat_tid = sol.threat_score.at(tid);
		// Cost of removing all k bursts from (wid, tid): objective rises by this amount
		double remove_cost = threat_tid * surv_tid * (std::pow(1.0 / (1.0 - p), k) - 1.0);

		for (const auto& [ck, slots] : sol.cap) {
			if (slots <= 0) continue;
			int w2 = static_cast<int>(ck >> 32);
			int t2 = static_cast<int>(ck  & 0xFFFFFFFF);
			if (w2 != wid || t2 == tid) continue;

			double p2 = detail::pair_p(sol, wid, t2);
			if (p2 <= 0.0) continue;
			double surv_t2   = sol.survival(t2);
			double threat_t2 = sol.threat_score.at(t2);
			// Gain from placing k bursts on (wid, t2): objective drops by this amount
			double gain  = threat_t2 * surv_t2 * (1.0 - std::pow(1.0 - p2, k));
			double delta = remove_cost - gain; // negative = net improvement

			if (delta < -1e-12)
				candidates.push_back({wid, tid, t2, k, delta});
		}
	}

	if (candidates.empty()) return false;

	// Sort: most analytically promising first
	std::sort(candidates.begin(), candidates.end(),
		[](const Move& a, const Move& b) { return a.delta < b.delta; });

	// Verify top candidates with an actual copy until we find a genuine improvement
	constexpr int MAX_TRIALS = 5;
	for (int i = 0; i < std::min<int>(MAX_TRIALS, static_cast<int>(candidates.size())); ++i) {
		const Move& mv = candidates[i];
		auto it = sol.fire_times.find(pair_key(mv.wid, mv.tid));
		if (it == sol.fire_times.end() || it->second.empty()) continue;
		std::vector<double> orig_times(it->second);

		Solution trial = sol.copy();
		for (double t : orig_times) trial.uncommit(mv.wid, mv.tid, t);

		int placed = 0;
		for (int s = 0; s < mv.k; ++s) {
			double t = trial.first_slot(mv.wid, mv.tid2);
			if (!detail::is_feasible_commit(trial, mv.wid, mv.tid2, t)) break;
			trial.commit(mv.wid, mv.tid2, t);
			++placed;
		}
		if (placed > 0 && trial.objective() < current_obj - 1e-12) {
			sol = std::move(trial);
			return true;
		}
	}
	return false;
}

// L2: best-improvement pair swap — scans ALL active (wid_a,tid_a)×(wid_b,tid_b) combinations
// analytically to find the cross-reassignment with the greatest objective drop.
// No same-weapon-type restriction; feasibility is enforced via cap and is_feasible_commit.
inline bool apply_swap_pair_best(
	Solution& sol,
	std::mt19937& /*rng*/,
	const ApplyParams& /*params*/)
{
	double current_obj = sol.objective();

	std::vector<uint64_t> pairs;
	for (const auto& [key, times] : sol.fire_times)
		if (!times.empty()) pairs.push_back(key);
	if (pairs.size() < 2) return false;

	struct Swap { int wid_a, tid_a, wid_b, tid_b; double delta; };
	std::vector<Swap> candidates;

	for (int i = 0; i < static_cast<int>(pairs.size()); ++i) {
		int wid_a = static_cast<int>(pairs[i] >> 32);
		int tid_a = static_cast<int>(pairs[i] & 0xFFFFFFFF);
		int k_a   = static_cast<int>(sol.fire_times.at(pairs[i]).size());
		if (k_a == 0) continue;

		double p_a      = detail::pair_p(sol, wid_a, tid_a);
		double surv_a   = sol.survival(tid_a);
		double threat_a = sol.threat_score.at(tid_a);
		if (p_a <= 0.0 || p_a >= 1.0) continue;

		for (int j = i + 1; j < static_cast<int>(pairs.size()); ++j) {
			int wid_b = static_cast<int>(pairs[j] >> 32);
			int tid_b = static_cast<int>(pairs[j] & 0xFFFFFFFF);
			if (wid_a == wid_b || tid_a == tid_b) continue;
			int k_b = static_cast<int>(sol.fire_times.at(pairs[j]).size());
			if (k_b == 0) continue;

			// Both cross-pairs must be feasible
			if (!sol.cap.count(pair_key(wid_a, tid_b))) continue;
			if (!sol.cap.count(pair_key(wid_b, tid_a))) continue;

			double p_b      = detail::pair_p(sol, wid_b, tid_b);
			double surv_b   = sol.survival(tid_b);
			double threat_b = sol.threat_score.at(tid_b);
			if (p_b <= 0.0 || p_b >= 1.0) continue;

			double p_ab = detail::pair_p(sol, wid_a, tid_b);
			double p_ba = detail::pair_p(sol, wid_b, tid_a);
			if (p_ab <= 0.0 || p_ba <= 0.0) continue;

			// Analytical delta: cost of removing both pairs minus gain from cross-placing
			double cost_a  = threat_a * surv_a * (std::pow(1.0 / (1.0 - p_a),  k_a) - 1.0);
			double cost_b  = threat_b * surv_b * (std::pow(1.0 / (1.0 - p_b),  k_b) - 1.0);
			double gain_ab = threat_b * surv_b * (1.0 - std::pow(1.0 - p_ab, k_a));
			double gain_ba = threat_a * surv_a * (1.0 - std::pow(1.0 - p_ba, k_b));
			double delta   = (cost_a + cost_b) - (gain_ab + gain_ba);

			if (delta < -1e-12)
				candidates.push_back({wid_a, tid_a, wid_b, tid_b, delta});
		}
	}

	if (candidates.empty()) return false;

	std::sort(candidates.begin(), candidates.end(),
		[](const Swap& a, const Swap& b) { return a.delta < b.delta; });

	constexpr int MAX_TRIALS = 5;
	for (int i = 0; i < std::min<int>(MAX_TRIALS, static_cast<int>(candidates.size())); ++i) {
		const Swap& sw = candidates[i];
		auto it_a = sol.fire_times.find(pair_key(sw.wid_a, sw.tid_a));
		auto it_b = sol.fire_times.find(pair_key(sw.wid_b, sw.tid_b));
		if (it_a == sol.fire_times.end() || it_b == sol.fire_times.end()) continue;
		std::vector<double> ta(it_a->second), tb(it_b->second);
		if (ta.empty() || tb.empty()) continue;

		Solution trial = sol.copy();
		for (double t : ta) trial.uncommit(sw.wid_a, sw.tid_a, t);
		for (double t : tb) trial.uncommit(sw.wid_b, sw.tid_b, t);

		int placed_a = 0;
		for (int s = 0; s < static_cast<int>(ta.size()); ++s) {
			double t = trial.first_slot(sw.wid_a, sw.tid_b);
			if (!detail::is_feasible_commit(trial, sw.wid_a, sw.tid_b, t)) break;
			trial.commit(sw.wid_a, sw.tid_b, t);
			++placed_a;
		}
		int placed_b = 0;
		for (int s = 0; s < static_cast<int>(tb.size()); ++s) {
			double t = trial.first_slot(sw.wid_b, sw.tid_a);
			if (!detail::is_feasible_commit(trial, sw.wid_b, sw.tid_a, t)) break;
			trial.commit(sw.wid_b, sw.tid_a, t);
			++placed_b;
		}
		if ((placed_a > 0 || placed_b > 0) && trial.objective() < current_obj - 1e-12) {
			sol = std::move(trial);
			return true;
		}
	}
	return false;
}

inline bool apply_llh(
	LLHId llh,
	Solution& sol,
	std::mt19937& rng,
	const ApplyParams& params = {})
{
	switch (llh) {
		case LLHId::M2_RUIN_RANDOM__H_SURV_THREAT_TIE:
			return apply_macro_move({RuinRule::RANDOM_PAIR, portfolio::HeuristicId::H_SURV_THREAT_TIE}, sol, rng, params);
		case LLHId::M4_RUIN_OVER_COVERED__H_EXCLUSIVE_RESERVE:
			return apply_macro_move({RuinRule::OVER_COVERED, portfolio::HeuristicId::H_EXCLUSIVE_RESERVE}, sol, rng, params);
		case LLHId::M6_RUIN_CONGESTED__H_WINDOW_CLOSURE:
			return apply_macro_move({RuinRule::CONGESTED, portfolio::HeuristicId::H_WINDOW_CLOSURE}, sol, rng, params);
		case LLHId::M7_RUIN_LOW_MARGINAL__H_COVER_FIRST:
			return apply_macro_move({RuinRule::LOW_MARGINAL_PAIR, portfolio::HeuristicId::H_COVER_FIRST}, sol, rng, params);
		case LLHId::M8_RUIN_CONGESTED__H_BACKLOG_RELIEF:
			return apply_macro_move({RuinRule::CONGESTED, portfolio::HeuristicId::H_BACKLOG_RELIEF}, sol, rng, params);
		case LLHId::M10_RUIN_RANDOM__H_SPREAD_THEN_FOCUS:
			return apply_macro_move({RuinRule::RANDOM_PAIR, portfolio::HeuristicId::H_SPREAD_THEN_FOCUS}, sol, rng, params);
		case LLHId::M11_RUIN_CONGESTED__H_ANTI_BOTTLENECK:
			return apply_macro_move({RuinRule::CONGESTED, portfolio::HeuristicId::H_ANTI_BOTTLENECK}, sol, rng, params);
		case LLHId::M12_RUIN_OVER_COVERED__H_OPPORTUNITY_LOCK:
			return apply_macro_move({RuinRule::OVER_COVERED, portfolio::HeuristicId::H_OPPORTUNITY_LOCK}, sol, rng, params);
		case LLHId::M13_RUIN_LOW_MARGINAL__H_MARGINAL_OBJECTIVE_DROP:
			return apply_macro_move({RuinRule::LOW_MARGINAL_PAIR, portfolio::HeuristicId::H_MARGINAL_OBJECTIVE_DROP}, sol, rng, params);
		case LLHId::M14_RUIN_OVER_COVERED__H_KILL_CHAIN_FINISHER:
			return apply_macro_move({RuinRule::OVER_COVERED, portfolio::HeuristicId::H_KILL_CHAIN_FINISHER}, sol, rng, params);
		case LLHId::M15_RUIN_CONGESTED__H_FUTURE_FLEX_PRESERVER:
			return apply_macro_move({RuinRule::CONGESTED, portfolio::HeuristicId::H_FUTURE_FLEX_PRESERVER}, sol, rng, params);
		case LLHId::M16_RUIN_RANDOM__H_BASELINE_TWO_STAGE:
			return apply_macro_move({RuinRule::RANDOM_PAIR, portfolio::HeuristicId::H_BASELINE_TWO_STAGE}, sol, rng, params);

		case LLHId::L1_REASSIGN_BEST_DELTA:
			return apply_reassign_pair_best(sol, rng, params);
		case LLHId::L2_SWAP_BEST_PAIR:
			return apply_swap_pair_best(sol, rng, params);
	}
	throw std::logic_error("Unhandled LLHId in apply_llh");
}

} // namespace perturbation