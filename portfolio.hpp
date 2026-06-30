#pragma once

#include "solution.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace portfolio {

enum class HeuristicId {
	H_SURV,
	H_SURV_THREAT_TIE,
	H_EXCLUSIVE_GUARD,
	H_WINDOW_URGENT,
	H_COVER_FIRST,
	H_EXCLUSIVE_RESERVE,
	H_WINDOW_CLOSURE,
	H_BACKLOG_RELIEF,
	H_FINISH_STARTED,
	H_SPREAD_THEN_FOCUS,
	H_ANTI_BOTTLENECK,
	H_OPPORTUNITY_LOCK,
	H_MARGINAL_OBJECTIVE_DROP,
	H_KILL_CHAIN_FINISHER,
	H_FUTURE_FLEX_PRESERVER,
	H_BASELINE_TWO_STAGE,
};

struct Candidate {
	int wid = -1;
	int tid = -1;
	double t = std::numeric_limits<double>::infinity();
};

struct SelectContext {
	double phase = 0.0;
	const std::unordered_map<int, int>* prev_target_feasible_weapons = nullptr;
};

struct SelectResult {
	bool found = false;
	Candidate cand;
};

namespace detail {

static inline bool better_surv(const Solution& sol, const Candidate& a, const Candidate& b) {
	constexpr double eps = 1e-12;
	double sa = sol.survival(a.tid);
	double sb = sol.survival(b.tid);
	if (sa > sb + eps) return true;
	if (sb > sa + eps) return false;
	if (a.t < b.t - eps) return true;
	if (b.t < a.t - eps) return false;
	if (a.wid < b.wid) return true;
	if (b.wid < a.wid) return false;
	return a.tid < b.tid;
}

static inline bool better_surv_then_threat(const Solution& sol, const Candidate& a, const Candidate& b) {
	constexpr double eps = 1e-12;
	double sa = sol.survival(a.tid);
	double sb = sol.survival(b.tid);
	if (sa > sb + eps) return true;
	if (sb > sa + eps) return false;
	double wa = sol.threat_score.at(a.tid);
	double wb = sol.threat_score.at(b.tid);
	if (wa > wb + eps) return true;
	if (wb > wa + eps) return false;
	if (a.t < b.t - eps) return true;
	if (b.t < a.t - eps) return false;
	if (a.wid < b.wid) return true;
	if (b.wid < a.wid) return false;
	return a.tid < b.tid;
}

static inline std::unordered_map<int, int>
count_target_feasible_weapons(const std::vector<Candidate>& candidates) {
	std::unordered_map<int, int> cnt;
	for (const auto& c : candidates) cnt[c.tid]++;
	return cnt;
}

static inline std::unordered_map<int, int>
count_weapon_feasible_actions(const std::vector<Candidate>& candidates) {
	std::unordered_map<int, int> cnt;
	for (const auto& c : candidates) cnt[c.wid]++;
	return cnt;
}

static inline const std::unordered_map<int, int>&
count_target_engagement(const Solution& sol) {
	return sol.engaged_count;
}

static inline bool target_is_exclusive(
	int tid,
	const std::unordered_map<int, int>& feasible_weapon_count
) {
	auto it = feasible_weapon_count.find(tid);
	return (it != feasible_weapon_count.end() && it->second == 1);
}

static inline bool weapon_has_any_exclusive_target(
	int wid,
	const std::vector<Candidate>& candidates,
	const std::unordered_map<int, int>& target_feasible_weapons
) {
	for (const auto& c : candidates) {
		if (c.wid != wid) continue;
		if (target_is_exclusive(c.tid, target_feasible_weapons)) return true;
	}
	return false;
}

static inline double slack_for(const Solution& sol, const Candidate& c) {
	auto it = sol.windows->find(pair_key(c.wid, c.tid));
	if (it == sol.windows->end()) return std::numeric_limits<double>::infinity();
	return it->second.second - c.t;
}

static inline double marginal_objective_drop(const Solution& sol, const Candidate& c) {
	auto pit = sol.p_ij->find(pair_key(c.wid, c.tid));
	double p = (pit == sol.p_ij->end()) ? 0.0 : pit->second;
	double surv = sol.survival(c.tid);
	double threat = sol.threat_score.at(c.tid);
	return surv * threat * p;
}

}

static inline SelectResult select_candidate(
	HeuristicId h,
	const Solution& sol,
	const std::vector<Candidate>& candidates,
	const SelectContext& ctx = {}
) {
	constexpr double eps = 1e-12;
	SelectResult out;
	if (candidates.empty()) return out;

	const auto target_feasible_weapons = detail::count_target_feasible_weapons(candidates);
	const auto weapon_action_count = detail::count_weapon_feasible_actions(candidates);
	const auto& target_engaged_count = detail::count_target_engagement(sol);

	auto choose_global = [&](auto better) {
		Candidate best = candidates.front();
		for (size_t i = 1; i < candidates.size(); ++i)
			if (better(candidates[i], best)) best = candidates[i];
		out.found = true;
		out.cand = best;
	};

	switch (h) {
		case HeuristicId::H_SURV:
			{
				constexpr double rho = 0.05;
				std::unordered_map<int, Candidate> best_by_weapon;
				std::unordered_map<int, double> best_sc_by_weapon;
				std::unordered_map<int, double> second_sc_by_weapon;

				for (const auto& c : candidates) {
					double sc = sol.survival(c.tid);
					auto it = best_by_weapon.find(c.wid);
					if (it == best_by_weapon.end()) {
						best_by_weapon[c.wid] = c;
						best_sc_by_weapon[c.wid] = sc;
						second_sc_by_weapon[c.wid] = -std::numeric_limits<double>::infinity();
						continue;
					}

					const Candidate& cur_best = it->second;
					double cur_best_sc = best_sc_by_weapon[c.wid];
					if (sc > cur_best_sc + eps ||
						(std::fabs(sc - cur_best_sc) <= eps && detail::better_surv(sol, c, cur_best))) {
						second_sc_by_weapon[c.wid] = cur_best_sc;
						best_by_weapon[c.wid] = c;
						best_sc_by_weapon[c.wid] = sc;
					} else if (sc > second_sc_by_weapon[c.wid] + eps) {
						second_sc_by_weapon[c.wid] = sc;
					}
				}

				bool have = false;
				Candidate chosen;
				double chosen_priority = -std::numeric_limits<double>::infinity();
				double chosen_best_sc = -std::numeric_limits<double>::infinity();

				for (const auto& [wid, best_c] : best_by_weapon) {
					double best_sc = best_sc_by_weapon[wid];
					double second_sc = second_sc_by_weapon[wid];
					if (!std::isfinite(second_sc)) second_sc = 0.0;
					double regret = best_sc - second_sc;
					double priority = regret + rho * best_sc;

					if (!have ||
						priority > chosen_priority + eps ||
						(std::fabs(priority - chosen_priority) <= eps &&
						 (best_sc > chosen_best_sc + eps ||
						  (std::fabs(best_sc - chosen_best_sc) <= eps &&
						   detail::better_surv(sol, best_c, chosen))))) {
						have = true;
						chosen = best_c;
						chosen_priority = priority;
						chosen_best_sc = best_sc;
					}
				}

				if (!have) return out;
				out.found = true;
				out.cand = chosen;
				return out;
			}

		case HeuristicId::H_SURV_THREAT_TIE:
			choose_global([&](const Candidate& a, const Candidate& b) {
				return detail::better_surv_then_threat(sol, a, b);
			});
			return out;

		case HeuristicId::H_EXCLUSIVE_GUARD:
		case HeuristicId::H_EXCLUSIVE_RESERVE: {
			std::vector<Candidate> filtered;
			filtered.reserve(candidates.size());
			for (const auto& c : candidates) {
				bool has_exclusive = detail::weapon_has_any_exclusive_target(
					c.wid, candidates, target_feasible_weapons);
				if (!has_exclusive || detail::target_is_exclusive(c.tid, target_feasible_weapons))
					filtered.push_back(c);
			}
			if (filtered.empty()) return out;
			Candidate best = filtered.front();
			for (size_t i = 1; i < filtered.size(); ++i) {
				if (detail::better_surv(sol, filtered[i], best)) best = filtered[i];
			}
			out.found = true;
			out.cand = best;
			return out;
		}

		case HeuristicId::H_WINDOW_URGENT:
		case HeuristicId::H_WINDOW_CLOSURE:
			choose_global([&](const Candidate& a, const Candidate& b) {
				double sa = detail::slack_for(sol, a);
				double sb = detail::slack_for(sol, b);
				if (sa < sb - eps) return true;
				if (sb < sa - eps) return false;
				if (a.t < b.t - eps) return true;
				if (b.t < a.t - eps) return false;
				return detail::better_surv_then_threat(sol, a, b);
			});
			return out;

		case HeuristicId::H_COVER_FIRST:
			choose_global([&](const Candidate& a, const Candidate& b) {
				int ca = target_feasible_weapons.at(a.tid);
				int cb = target_feasible_weapons.at(b.tid);
				if (ca < cb) return true;
				if (cb < ca) return false;
				int wa = weapon_action_count.at(a.wid);
				int wb = weapon_action_count.at(b.wid);
				if (wa > wb) return true;
				if (wb > wa) return false;
				return detail::better_surv_then_threat(sol, a, b);
			});
			return out;

		case HeuristicId::H_BACKLOG_RELIEF: {
			int chosen_w = candidates.front().wid;
			int best_load = weapon_action_count.at(chosen_w);
			for (const auto& [wid, cnt] : weapon_action_count) {
				if (cnt > best_load || (cnt == best_load && wid < chosen_w)) {
					chosen_w = wid;
					best_load = cnt;
				}
			}
			bool have = false;
			Candidate best;
			for (const auto& c : candidates) {
				if (c.wid != chosen_w) continue;
				if (!have || detail::better_surv(sol, c, best)) {
					best = c;
					have = true;
				}
			}
			if (!have) return out;
			out.found = true;
			out.cand = best;
			return out;
		}

		case HeuristicId::H_FINISH_STARTED:
			choose_global([&](const Candidate& a, const Candidate& b) {
				int ea = (target_engaged_count.count(a.tid) ? 1 : 0);
				int eb = (target_engaged_count.count(b.tid) ? 1 : 0);
				if (ea > eb) return true;
				if (eb > ea) return false;
				return detail::better_surv(sol, a, b);
			});
			return out;

		case HeuristicId::H_SPREAD_THEN_FOCUS:
			if (ctx.phase < 0.33) {
				choose_global([&](const Candidate& a, const Candidate& b) {
					int ua = (target_engaged_count.count(a.tid) ? 0 : 1);
					int ub = (target_engaged_count.count(b.tid) ? 0 : 1);
					if (ua > ub) return true;
					if (ub > ua) return false;
					return detail::better_surv(sol, a, b);
				});
				return out;
			}
			if (ctx.phase < 0.66) {
				choose_global([&](const Candidate& a, const Candidate& b) {
					return detail::better_surv(sol, a, b);
				});
				return out;
			}
			choose_global([&](const Candidate& a, const Candidate& b) {
				double sa = detail::slack_for(sol, a);
				double sb = detail::slack_for(sol, b);
				if (sa < sb - eps) return true;
				if (sb < sa - eps) return false;
				return detail::better_surv(sol, a, b);
			});
			return out;

		case HeuristicId::H_ANTI_BOTTLENECK: {
			std::unordered_map<int, double> min_t_by_weapon;
			for (const auto& c : candidates) {
				auto it = min_t_by_weapon.find(c.wid);
				if (it == min_t_by_weapon.end()) min_t_by_weapon[c.wid] = c.t;
				else it->second = std::min(it->second, c.t);
			}
			int chosen_w = candidates.front().wid;
			double best_min_t = min_t_by_weapon.at(chosen_w);
			for (const auto& [wid, min_t] : min_t_by_weapon) {
				if (min_t > best_min_t + eps ||
					(std::fabs(min_t - best_min_t) <= eps && wid < chosen_w)) {
					chosen_w = wid;
					best_min_t = min_t;
				}
			}
			bool have = false;
			Candidate best;
			for (const auto& c : candidates) {
				if (c.wid != chosen_w) continue;
				if (!have) {
					best = c;
					have = true;
					continue;
				}
				double sc = detail::slack_for(sol, c);
				double sb = detail::slack_for(sol, best);
				if (sc < sb - eps ||
					(std::fabs(sc - sb) <= eps && detail::better_surv(sol, c, best))) {
					best = c;
				}
			}
			if (!have) return out;
			out.found = true;
			out.cand = best;
			return out;
		}

		case HeuristicId::H_OPPORTUNITY_LOCK:
			choose_global([&](const Candidate& a, const Candidate& b) {
				int cur_a = target_feasible_weapons.at(a.tid);
				int cur_b = target_feasible_weapons.at(b.tid);
				int prev_a = cur_a;
				int prev_b = cur_b;
				if (ctx.prev_target_feasible_weapons) {
					auto ia = ctx.prev_target_feasible_weapons->find(a.tid);
					auto ib = ctx.prev_target_feasible_weapons->find(b.tid);
					if (ia != ctx.prev_target_feasible_weapons->end()) prev_a = ia->second;
					if (ib != ctx.prev_target_feasible_weapons->end()) prev_b = ib->second;
				}
				int drop_a = prev_a - cur_a;
				int drop_b = prev_b - cur_b;
				if (drop_a > drop_b) return true;
				if (drop_b > drop_a) return false;
				if (a.t < b.t - eps) return true;
				if (b.t < a.t - eps) return false;
				return detail::better_surv_then_threat(sol, a, b);
			});
			return out;

		case HeuristicId::H_MARGINAL_OBJECTIVE_DROP:
			choose_global([&](const Candidate& a, const Candidate& b) {
				double da = detail::marginal_objective_drop(sol, a);
				double db = detail::marginal_objective_drop(sol, b);
				if (da > db + eps) return true;
				if (db > da + eps) return false;
				if (a.t < b.t - eps) return true;
				if (b.t < a.t - eps) return false;
				return detail::better_surv_then_threat(sol, a, b);
			});
			return out;

		case HeuristicId::H_KILL_CHAIN_FINISHER: {
			std::vector<Candidate> engaged;
			engaged.reserve(candidates.size());
			for (const auto& c : candidates) {
				if (target_engaged_count.count(c.tid)) engaged.push_back(c);
			}
			const std::vector<Candidate>& pool = engaged.empty() ? candidates : engaged;

			Candidate best = pool.front();
			for (size_t i = 1; i < pool.size(); ++i) {
				const Candidate& c = pool[i];
				double dc = detail::marginal_objective_drop(sol, c);
				double db = detail::marginal_objective_drop(sol, best);
				if (dc > db + eps) {
					best = c;
					continue;
				}
				if (db > dc + eps) continue;

				int fc = target_feasible_weapons.at(c.tid);
				int fb = target_feasible_weapons.at(best.tid);
				if (fc < fb) {
					best = c;
					continue;
				}
				if (fb < fc) continue;

				if (c.t < best.t - eps) {
					best = c;
					continue;
				}
				if (best.t < c.t - eps) continue;

				if (detail::better_surv_then_threat(sol, c, best)) {
					best = c;
				}
			}

			out.found = true;
			out.cand = best;
			return out;
		}

		case HeuristicId::H_FUTURE_FLEX_PRESERVER:
			choose_global([&](const Candidate& a, const Candidate& b) {
				constexpr double lambda = 0.15;
				double va = detail::marginal_objective_drop(sol, a)
					- lambda / (1.0 + static_cast<double>(weapon_action_count.at(a.wid)));
				double vb = detail::marginal_objective_drop(sol, b)
					- lambda / (1.0 + static_cast<double>(weapon_action_count.at(b.wid)));
				if (va > vb + eps) return true;
				if (vb > va + eps) return false;
				if (a.t < b.t - eps) return true;
				if (b.t < a.t - eps) return false;
				return detail::better_surv_then_threat(sol, a, b);
			});
			return out;

		case HeuristicId::H_BASELINE_TWO_STAGE: {
			int    sel_tid = -1;
			double sel_sc  = -std::numeric_limits<double>::infinity();
			double sel_th  = -std::numeric_limits<double>::infinity();
			double sel_we  =  std::numeric_limits<double>::infinity();

			for (const auto& c : candidates) {
				double sc = sol.survival(c.tid) * sol.threat_score.at(c.tid);
				double th = sol.threat_score.at(c.tid);
				auto w_it = sol.windows->find(pair_key(c.wid, c.tid));
				double we = (w_it == sol.windows->end())
					? std::numeric_limits<double>::infinity()
					: w_it->second.second;

				if (sel_tid < 0 ||
					sc > sel_sc + eps ||
					(std::fabs(sc - sel_sc) <= eps &&
					 (th > sel_th + eps ||
					  (std::fabs(th - sel_th) <= eps &&
					   (we < sel_we - eps ||
						(std::fabs(we - sel_we) <= eps && c.tid < sel_tid)))))) {
					sel_tid = c.tid;
					sel_sc  = sc;
					sel_th  = th;
					sel_we  = we;
				}
			}

			if (sel_tid < 0) return out;

			bool have = false;
			Candidate chosen;
			int best_scarcity = std::numeric_limits<int>::max();
			int best_excl     = -1;

			for (const auto& c : candidates) {
				if (c.tid != sel_tid) continue;
				int scarcity = weapon_action_count.at(c.wid);
				int excl_live = 0;
				for (const auto& c2 : candidates) {
					if (c2.wid != c.wid) continue;
					if (target_feasible_weapons.at(c2.tid) == 1) ++excl_live;
				}
				if (!have ||
					scarcity < best_scarcity ||
					(scarcity == best_scarcity &&
					 (excl_live > best_excl ||
					  (excl_live == best_excl &&
					   (c.t < chosen.t - eps ||
						(std::fabs(c.t - chosen.t) <= eps && c.wid < chosen.wid)))))) {
					have          = true;
					chosen        = c;
					best_scarcity = scarcity;
					best_excl     = excl_live;
				}
			}

			if (!have) return out;
			out.found = true;
			out.cand  = chosen;
			return out;
		}
	}

	return out;
}

}