#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "heuristic.hpp"
#include "perturbation.hpp"

using json = nlohmann::json;

struct Scenario {
	std::vector<Weapon>  weapons;
	std::vector<Target>  targets;
	std::unordered_map<uint64_t, double>                   p_ij;
	std::unordered_map<uint64_t, std::pair<double,double>> windows;
	std::unordered_map<int, double>                        burst_dur;
	std::unordered_map<int, int>                           max_shots;
	std::unordered_map<int, int>                           vessel_id_map;
	double horizon = 60.0;
};

static Scenario load_scenario(const std::string& path) {
	std::ifstream f(path);
	if (!f) throw std::runtime_error("cannot open: " + path);
	json data = json::parse(f);

	Scenario sc;

	std::unordered_map<std::string, WeaponInfo> winfo_map;
	for (auto& item : data["weapon_infos"]) {
		WeaponInfo wi;
		wi.id                   = item["ID"];
		wi.code                 = item["Code"];
		wi.type                 = item["Type"];
		wi.min_range            = item["MinRange"];
		wi.max_range            = item["MaxRange"];
		wi.min_altitude         = item["MinAltitude"];
		wi.max_altitude         = item["MaxAltitude"];
		wi.azimuth_from_deg     = item["AzimuthFromDeg"];
		wi.azimuth_to_deg       = item["AzimuthToDeg"];
		wi.elevation_min_deg    = item["ElevationMinDeg"];
		wi.elevation_max_deg    = item["ElevationMaxDeg"];
		wi.max_shots_per_target = item["MaxShotsPerTarget"];
		wi.rounds_per_burst     = item["RoundsPerBurst"];
		wi.burst_interval       = item["BurstInterval"];
		wi.reload_time          = item["ReloadTime"];
		winfo_map[wi.code]      = wi;
	}

	std::unordered_map<std::string, TargetInfo> tinfo_map;
	for (auto& item : data["target_infos"]) {
		TargetInfo ti;
		ti.id          = item["ID"];
		ti.code        = item["Code"];
		ti.description = item.value("Description", "");
		ti.type        = item["Type"];
		tinfo_map[ti.code] = ti;
	}

	std::unordered_map<std::string, double> prob_map;
	for (auto& row : data["probability_table"]) {
		std::string key = std::string(row["WTAWeaponInfoCode"]) + "|"
						+ std::string(row["WTATargetInfoCode"]);
		prob_map[key] = row["Score"];
	}

	auto& req = data["assignment_request"];

	std::unordered_map<int, std::string> weapon_info_code;
	for (auto& item : req["weapons"]) {
		Weapon w;
		w.id        = item["ID"];
		w.vessel_id = item["WTAVesselID"];
		w.ammo      = item["Ammo"];
		w.info_code = item["WTAWeaponInfoCode"];
		w.status    = item["Status"];
		sc.weapons.push_back(w);

		const WeaponInfo& wi   = winfo_map.at(w.info_code);
		sc.burst_dur[w.id]     = wi.burst_duration();
		sc.max_shots[w.id]     = wi.max_shots_per_target;
		sc.vessel_id_map[w.id] = w.vessel_id;
		weapon_info_code[w.id] = w.info_code;
	}

	std::unordered_map<int, std::string> target_info_code;
	for (auto& item : req["targets"]) {
		Target t;
		t.id           = item["ID"];
		t.info_code    = item["WTATargetInfoCode"];
		t.x            = item["X"];
		t.y            = item["Y"];
		t.z            = item["Z"];
		t.vx           = item.value("VX", 0.0);
		t.vy           = item.value("VY", 0.0);
		t.vz           = item.value("VZ", 0.0);
		t.speed        = item["Speed"];
		t.threat_score = item["ThreatScore"];
		sc.targets.push_back(t);
		target_info_code[t.id] = t.info_code;
	}

	for (auto& [key_str, ab] : data["engagement_windows"].items()) {
		auto sep = key_str.find('_');
		int wid = std::stoi(key_str.substr(0, sep));
		int tid = std::stoi(key_str.substr(sep + 1));

		double a = ab[0], b = ab[1];

		std::string pkey = weapon_info_code.at(wid) + "|" + target_info_code.at(tid);
		auto pit = prob_map.find(pkey);
		if (pit == prob_map.end() || pit->second <= 0.0) continue;

		uint64_t k = pair_key(wid, tid);
		const WeaponInfo& wi =
			winfo_map.at(weapon_info_code.at(wid));

		sc.windows[k] = {
			a,
			b + wi.reload_time
		};
		sc.p_ij[k]    = pit->second;
	}

	if (!sc.windows.empty()) {
		sc.horizon = 0.0;
		for (auto& [k, ab] : sc.windows)
			sc.horizon = std::max(sc.horizon, ab.second);
	}

	return sc;
}

static void write_solution(const Solution& sol, const std::string& path) {
	auto assignments = sol.assignments();
	std::sort(assignments.begin(), assignments.end(),
		[](const Assignment& a, const Assignment& b) {
			if (a.vessel_id != b.vessel_id) return a.vessel_id < b.vessel_id;
			if (a.weapon_id != b.weapon_id) return a.weapon_id < b.weapon_id;
			return a.target_id < b.target_id;
		});

	json out;
	out["objective"] = sol.objective();
	out["compactness"] = sol.compactness();
	out["lexicographic_objective"] = {
		{"primary", sol.objective()},
		{"secondary", sol.compactness()}
	};
	json arr = json::array();
	for (auto& a : assignments) {
		json rec = {
			{"WTAVesselID", a.vessel_id},
			{"WTAWeaponID", a.weapon_id},
			{"WTATargetID", a.target_id},
			{"AmmoUsed",    a.ammo_used},
			{"PKill",       a.pkill},
			{"FireTime",    a.fire_time},
			{"EndTime",     a.end_time}
		};
		rec["FireTimes"] = a.fire_times;
		arr.push_back(rec);
	}
	out["assignments"] = arr;

	std::ofstream f(path);
	if (!f) throw std::runtime_error("cannot write output: " + path);
	f << out.dump(2) << "\n";
	std::cout << "Solution written to " << path << "\n";
}

static double dynamic_beta(
    double progress_pct,
    double x_ini,
    double x_end,
    double y_ini,
    double y_end)
{
    if (std::fabs(x_end - x_ini) < 1e-12)
        return y_ini;

    double x = std::clamp(
        progress_pct,
        std::min(x_ini, x_end),
        std::max(x_ini, x_end)
    );

    double ratio =
        (x - x_ini) /
        (x_end - x_ini);

    return y_ini
         + ratio * (y_end - y_ini);
}

static bool lex_primary_equal(
    const LexScore& lhs,
    const LexScore& rhs)
{
    constexpr double eps = 1e-10;
    const double scale = std::max({
        1.0,
        std::fabs(lhs.primary),
        std::fabs(rhs.primary)
    });
    return std::fabs(lhs.primary - rhs.primary) <= eps * scale;
}

static double lex_worsening_energy(
    const LexScore& candidate,
    const LexScore& incumbent,
    double primary_scale,
    double compactness_scale)
{
    if (!lex_primary_equal(candidate, incumbent)) {
        return std::max(0.0, candidate.primary - incumbent.primary);
    }

    const double compactness_delta =
        std::max(0.0, candidate.compactness - incumbent.compactness);

    return compactness_delta
         / std::max(1e-12, compactness_scale)
         * std::max(1e-12, primary_scale);
}

// UCB Sampling
static int select_llh_ucb(
	const std::vector<double>& suc,
	const std::vector<int>& selected,
	int k,
	std::mt19937& /*rng*/)
{
	const int n = static_cast<int>(suc.size());
	for (int i = 0; i < n; ++i)
		if (selected[i] == 0) return i;
	double best = -std::numeric_limits<double>::infinity();
	int argbest = 0;
	const double lnk = std::log(static_cast<double>(std::max(1, k)));
	for (int i = 0; i < n; ++i) {
		double v = suc[i] / static_cast<double>(selected[i])
		         + std::sqrt(2.0 * lnk / static_cast<double>(selected[i]));
		if (v > best) { best = v; argbest = i; }
	}
	return argbest;
}

static int select_llh_ucb_enabled(
	const std::vector<double>& suc,
	const std::vector<int>& selected,
	const std::vector<char>& enabled,
	int k,
	std::mt19937& /*rng*/)
{
	const int n = static_cast<int>(suc.size());
	for (int i = 0; i < n; ++i)
		if (enabled[i] && selected[i] == 0) return i;

	double best = -std::numeric_limits<double>::infinity();
	int argbest = -1;
	const double lnk = std::log(static_cast<double>(std::max(1, k)));
	for (int i = 0; i < n; ++i) {
		if (!enabled[i]) continue;
		double v = suc[i] / static_cast<double>(std::max(1, selected[i]))
		         + std::sqrt(2.0 * lnk / static_cast<double>(std::max(1, selected[i])));
		if (v > best) { best = v; argbest = i; }
	}
	return argbest;
}

// Thompson Sampling
static int select_llh_thompson(
	const std::vector<double>& suc,
	std::mt19937& rng)
{
	const int n = static_cast<int>(suc.size());
	double best = -std::numeric_limits<double>::infinity();
	int argbest = 0;
	std::uniform_real_distribution<double> u01(std::numeric_limits<double>::min(), 1.0);
	for (int i = 0; i < n; ++i) {
		double a = suc[i] + 1.0;
		double theta = std::pow(u01(rng), 1.0 / std::max(1e-12, a));
		if (theta > best) { best = theta; argbest = i; }
	}
	return argbest;
}

static int select_llh_thompson_enabled(
	const std::vector<double>& suc,
	const std::vector<char>& enabled,
	std::mt19937& rng)
{
	const int n = static_cast<int>(suc.size());
	double best = -std::numeric_limits<double>::infinity();
	int argbest = -1;
	std::uniform_real_distribution<double> u01(std::numeric_limits<double>::min(), 1.0);
	for (int i = 0; i < n; ++i) {
		if (!enabled[i]) continue;
		double a = suc[i] + 1.0;
		double theta = std::pow(u01(rng), 1.0 / std::max(1e-12, a));
		if (theta > best) { best = theta; argbest = i; }
	}
	return argbest;
}

static int select_random_enabled(
	const std::vector<char>& enabled,
	std::mt19937& rng)
{
	std::vector<int> arms;
	arms.reserve(enabled.size());
	for (int i = 0; i < static_cast<int>(enabled.size()); ++i)
		if (enabled[i]) arms.push_back(i);
	if (arms.empty()) return -1;
	int pick = std::uniform_int_distribution<int>(0, static_cast<int>(arms.size()) - 1)(rng);
	return arms[pick];
}

static bool lightweight_feasible(const Solution& sol) {
	for (const auto& [wid, rem] : sol.remaining_ammo) {
		if (rem < 0) return false;
		(void)wid;
	}

	for (const auto& [key, shots] : sol.k) {
		if (shots < 0) return false;
		int wid = static_cast<int>(key >> 32);
		if (sol.max_shots && sol.max_shots->count(wid)) {
			if (shots > sol.max_shots->at(wid)) return false;
		}
	}

	for (const auto& [key, times] : sol.fire_times) {
		if (times.empty()) continue;
		int wid = static_cast<int>(key >> 32);
		auto w_it = sol.windows->find(key);
		if (w_it == sol.windows->end()) return false;
		double a = w_it->second.first;
		double b = w_it->second.second;
		double d = sol.burst_dur->at(wid);
		for (double t : times) {
			if (std::isnan(t)) return false;
			if (t < a - 1e-9) return false;
			if (t + d > b + 1e-9) return false;
		}
	}

	return true;
}

int main(int argc, char* argv[]) {
	std::string scenario_path = "/workspaces/WTA/data/scenario_001.json";
	std::string output_path;
	std::string config_path   = "config.json";

	int restarts = 1;
	double grasp_alpha = 0.85;
	uint32_t seed = 42;

	int    max_acc       = 0;
	int    iiter         = 0;
	int    runs          = 1;
	double search_seconds = 0.0;
	double temp_init     = -1.0;
	double cooling_alpha = 0.99;
	int    limit         = 20;
	double x_ini         = 0.0;
	double x_end         = 90.0;
	double y_ini         = 1.0;
	double y_end         = 0.05;
	int    rl_type       = 2;
	int    hard_reset_after_blocks = 6;

	// load config.json ("hhasa_rl" section) before parsing CLI args
	{
		std::ifstream cfg_f(config_path);
		if (cfg_f) {
			try {
				json cfg = json::parse(cfg_f);
				if (cfg.contains("hhasa_rl")) {
					auto& g = cfg["hhasa_rl"];
					if (g.contains("seed"))                 seed                 = g["seed"].get<uint32_t>();
					if (g.contains("restarts"))             restarts             = g["restarts"].get<int>();
					if (g.contains("grasp_alpha"))          grasp_alpha          = g["grasp_alpha"].get<double>();
					if (g.contains("runs"))                 runs                 = g["runs"].get<int>();
					if (g.contains("search_seconds"))       search_seconds       = g["search_seconds"].get<double>();
					if (g.contains("max_acc"))              max_acc              = g["max_acc"].get<int>();
					if (g.contains("iiter"))                iiter                = g["iiter"].get<int>();
					if (g.contains("temp_init"))            temp_init            = g["temp_init"].get<double>();
					if (g.contains("cooling_alpha"))        cooling_alpha        = g["cooling_alpha"].get<double>();
					if (g.contains("limit"))                limit                = g["limit"].get<int>();
					if (g.contains("x_ini"))                x_ini                = g["x_ini"].get<double>();
					if (g.contains("x_end"))                x_end                = g["x_end"].get<double>();
					if (g.contains("y_ini"))                y_ini                = g["y_ini"].get<double>();
					if (g.contains("y_end"))                y_end                = g["y_end"].get<double>();
					if (g.contains("rl_type"))              rl_type              = g["rl_type"].get<int>();
					if (g.contains("hard_reset_after_blocks")) hard_reset_after_blocks = g["hard_reset_after_blocks"].get<int>();
				}
			} catch (const std::exception& e) {
				std::cerr << "[config] parse error: " << e.what() << "\n";
			}
		}
	}

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--config"  && i + 1 < argc) config_path = argv[++i];
		else if (arg == "--restarts" && i + 1 < argc) restarts = std::stoi(argv[++i]);
		else if (arg == "--grasp-alpha" && i + 1 < argc) grasp_alpha = std::stod(argv[++i]);
		else if (arg == "--seed" && i + 1 < argc) seed = static_cast<uint32_t>(std::stoul(argv[++i]));
		else if (arg == "--macc" && i + 1 < argc) max_acc = std::stoi(argv[++i]);
		else if (arg == "--iiter" && i + 1 < argc) iiter = std::stoi(argv[++i]);
		else if (arg == "--temp" && i + 1 < argc) temp_init = std::stod(argv[++i]);
		else if (arg == "--cooling-alpha" && i + 1 < argc) cooling_alpha = std::stod(argv[++i]);
		else if (arg == "--limit" && i + 1 < argc) limit = std::stoi(argv[++i]);
		else if (arg == "--xini" && i + 1 < argc) x_ini = std::stod(argv[++i]);
		else if (arg == "--xend" && i + 1 < argc) x_end = std::stod(argv[++i]);
		else if (arg == "--yini" && i + 1 < argc) y_ini = std::stod(argv[++i]);
		else if (arg == "--yend" && i + 1 < argc) y_end = std::stod(argv[++i]);
		else if (arg == "--runs" && i + 1 < argc) runs        = std::stoi(argv[++i]);
		else if (arg == "--search-seconds" && i + 1 < argc) search_seconds = std::stod(argv[++i]);
		else if (arg == "--rl"   && i + 1 < argc) rl_type     = std::stoi(argv[++i]);
		else if (arg == "--hard-reset-blocks" && i + 1 < argc) hard_reset_after_blocks = std::stoi(argv[++i]);
		else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
		else if (arg[0] != '-') scenario_path = arg;
	}

	hard_reset_after_blocks = std::max(1, hard_reset_after_blocks);

	if (output_path.empty()) {
		output_path = scenario_path;
		auto pos = output_path.rfind('.');
		if (pos != std::string::npos) output_path.erase(pos);
		output_path += "_solution.json";
	}

	std::cout << "Loading " << scenario_path << " ...\n";
	Scenario sc = load_scenario(scenario_path);
	std::cout << "  weapons=" << sc.weapons.size()
			  << "  targets=" << sc.targets.size()
			  << "  pairs=" << sc.windows.size()
			  << "  horizon=" << sc.horizon << "s\n";

	const int nc = static_cast<int>(sc.targets.size());
	if (max_acc <= 0) max_acc = 25000 * nc;
	if (iiter   <= 0) iiter   = 40    * nc;

	std::cout << std::fixed << std::setprecision(6);
	std::cout << "  nc=" << nc
			  << "  macc=" << max_acc
			  << "  iiter=" << iiter
			  << "  search_seconds=" << search_seconds
			  << "  cooling_alpha=" << cooling_alpha
			  << "  limit=" << limit
			  << "  hard_reset_after_blocks=" << hard_reset_after_blocks
			  << "  rl=" << rl_type
			  << "  runs=" << runs << "\n";

	constexpr int N = perturbation::kLLHCount;

	Solution overall_best;
	LexScore overall_best_score{
		std::numeric_limits<double>::infinity(),
		std::numeric_limits<double>::infinity()
	};
	bool have_overall = false;

	std::vector<LexScore> run_results;
	run_results.reserve(runs);

	const auto wall_start = std::chrono::steady_clock::now();

	std::vector<char> llh_enabled(N, 1);
	int enabled_count = 0;
	for (char e : llh_enabled) enabled_count += (e ? 1 : 0);
	std::cout << "  active_llh=" << enabled_count << "/" << N << "\n";

	for (int run = 0; run < runs; ++run) {
		uint32_t run_seed = seed + static_cast<uint32_t>(run);
		std::mt19937 rng(run_seed);

		const auto run_start = std::chrono::steady_clock::now();

		std::unordered_map<int, LexScore> run_best_per_second;

		Solution s = grasp(
			sc.weapons, sc.targets, sc.p_ij, sc.windows,
			sc.burst_dur, sc.max_shots, sc.vessel_id_map,
			sc.horizon, grasp_alpha, restarts, run_seed);

		const LexScore score_init = lex_score(s);
		std::cout << std::fixed << std::setprecision(10)
				  << "[run " << (run + 1) << "/" << runs
				  << "  seed=" << run_seed
				  << "]  GRASP init score=(" << score_init.primary
				  << ", " << std::setprecision(4) << score_init.compactness
				  << ")\n";

		run_best_per_second[0] = score_init;

		Solution s_best = s.copy();
		LexScore score_best = score_init;
		LexScore score_cur_state = score_init;

		double primary_energy_scale =
			std::max(1.0, score_init.primary);
		double compactness_energy_scale = std::max({
			1.0,
			score_init.compactness,
			sc.horizon
		});

		double T0 = (temp_init > 0.0) ? temp_init
					                   : std::max(1e-6, 0.01 * primary_energy_scale);
		double T = T0;

		std::vector<double> val(N, 0.2);
		std::vector<int> usage(N, 0);
		std::vector<int> accepted_worse(N, 0);
		std::vector<int> improved_cnt(N, 0);
		std::vector<int> best_improve_cnt(N, 0);
		int best_improve_iters = 0;
		std::uniform_real_distribution<double> unit01(0.0, 1.0);

		int acc = 0;
		int hup = 0;
		int non_improving_blocks = 0;
		const bool use_time_budget = (search_seconds > 0.0);

		while (acc < max_acc) {
			if (use_time_budget) {
				double elapsed = std::chrono::duration<double>(
					std::chrono::steady_clock::now() - run_start).count();
				if (elapsed >= search_seconds) break;
			}
		// Per-block bandit state (reset after each inner loop, matching EVRPSARL.m)
		std::vector<double> suc(N, 0.0);    
		std::vector<int>    pulls(N, 0);    
		int total_pulls = 0;
		std::vector<char> llh_enabled_block = llh_enabled;

		Solution block_best;
		bool block_best_updated = false;
		LexScore score_block_best = score_cur_state;
		double progress_time_ratio = 0.0;
		int clock_tick = 0;

		for (int k = 0; k < iiter && acc < max_acc; ++k) {
			if (use_time_budget && ((clock_tick++ & 15) == 0)) {
				double elapsed = std::chrono::duration<double>(
					std::chrono::steady_clock::now() - run_start).count();
				progress_time_ratio = std::clamp(elapsed / std::max(1e-9, search_seconds), 0.0, 1.0);
				if (elapsed >= search_seconds) break;
			}
			int heur;
			if (rl_type == 3) {
				heur = select_llh_ucb_enabled(suc, pulls, llh_enabled_block, std::max(1, total_pulls), rng);
			} else if (rl_type == 2) {
				heur = select_llh_thompson_enabled(suc, llh_enabled_block, rng);
			} else {
				// rl_type 0: random
				heur = select_random_enabled(llh_enabled_block, rng);
			}
			if (heur < 0) break;
			usage[heur]++;

			Solution cand = s.copy();
			bool changed = perturbation::apply_llh(
				static_cast<perturbation::LLHId>(heur),
				cand,
				rng,
				perturbation::ApplyParams{val[heur]});

			if (!changed) {
				pulls[heur]++;
				total_pulls++;
				continue;
			}

			if (!lightweight_feasible(cand)) {
				bool repaired = perturbation::apply_llh(
					perturbation::LLHId::L1_REASSIGN_BEST_DELTA,
					cand,
					rng,
					perturbation::ApplyParams{1.0});
				if (repaired && !lightweight_feasible(cand)) {
					repaired = perturbation::apply_llh(
						perturbation::LLHId::L2_SWAP_BEST_PAIR,
						cand,
						rng,
						perturbation::ApplyParams{1.0});
				}
				if (!repaired || !lightweight_feasible(cand)) {
					pulls[heur]++;
					total_pulls++;
					continue;
				}
			}

			const LexScore score_cand = lex_score(cand);
			const LexRelation relation_to_current =
				lex_relation(score_cand, score_cur_state);

			const LexScore best_reference =
				(block_best_updated &&
				 lex_better(score_block_best, score_best))
				? score_block_best
				: score_best;

			const bool is_new_run_best =
				lex_better(score_cand, best_reference);

			acc++;

			bool accept = false;
			if (relation_to_current == LexRelation::Better) {
				accept = true;
				suc[heur] += 1.0;
				improved_cnt[heur]++;
				if (is_new_run_best) {
					best_improve_cnt[heur]++;
				}
			} else if (relation_to_current == LexRelation::Equal) {
				accept = true;
			} else {
				const double worsening_energy =
					lex_worsening_energy(
						score_cand,
						score_cur_state,
						primary_energy_scale,
						compactness_energy_scale);

				const double p = std::exp(
					-worsening_energy / std::max(1e-12, T));

				if (unit01(rng) < p) {
					accept = true;
					accepted_worse[heur]++;
				}
			}

			pulls[heur]++;
			total_pulls++;

			if (accept) {
				s = std::move(cand);
				score_cur_state = score_cand;

				if (lex_better(score_cur_state, score_block_best)) {
					block_best = s.copy();
					score_block_best = score_cur_state;
					block_best_updated = true;
				}
			}

			int reward_case = 3;
			// 0: new run best, 1: improve current, 2: worsen current, 3: equal
			if (relation_to_current == LexRelation::Better) {
				reward_case = is_new_run_best ? 0 : 1;
			} else if (relation_to_current == LexRelation::Worse) {
				reward_case = 2;
			}

			static constexpr double diff[4] = {
				0.0025, 0.00025, -0.00025, -0.00005
			};
			val[heur] = std::clamp(
				val[heur] + diff[reward_case],
				0.2,
				1.0);

			// Record the cumulative lexicographic best found by this second.
			{
				double elapsed = std::chrono::duration<double>(
					std::chrono::steady_clock::now() - run_start).count();
				int sec = static_cast<int>(elapsed);

				if (sec == 0) {
					continue;
				}

				const LexScore best_so_far =
					(block_best_updated &&
					 lex_better(score_block_best, score_best))
					? score_block_best
					: score_best;

				auto it = run_best_per_second.find(sec);
				if (it == run_best_per_second.end()) {
					run_best_per_second[sec] = best_so_far;
				} else if (lex_better(best_so_far, it->second)) {
					it->second = best_so_far;
				}
			}
		}

			if (block_best_updated &&
				lex_better(score_block_best, score_best)) {
				s_best = std::move(block_best);
				score_best = score_block_best;
				best_improve_iters += 1;
				hup = 0;
				non_improving_blocks = 0;
			} else {
				hup += 1;
				non_improving_blocks += 1;
			}

			if (non_improving_blocks >= hard_reset_after_blocks) {
				uint32_t reset_seed = run_seed
					+ static_cast<uint32_t>(acc)
					+ static_cast<uint32_t>(best_improve_iters)
					+ static_cast<uint32_t>(non_improving_blocks)
					+ 1u;

				Solution reset_sol = grasp(
					sc.weapons, sc.targets, sc.p_ij, sc.windows,
					sc.burst_dur, sc.max_shots, sc.vessel_id_map,
					sc.horizon, grasp_alpha, restarts, reset_seed);

				s = reset_sol.copy();
				score_cur_state = lex_score(s);

				primary_energy_scale = std::max(
					primary_energy_scale,
					std::max(1.0, score_cur_state.primary));
				compactness_energy_scale = std::max({
					compactness_energy_scale,
					score_cur_state.compactness,
					sc.horizon
				});

				double reset_T0 = (temp_init > 0.0)
					? temp_init
					: std::max(1e-6, 0.01 * primary_energy_scale);
				T = reset_T0;

				val.assign(N, 0.2);
				hup = 0;
				non_improving_blocks = 0;

				if (lex_better(score_cur_state, score_best)) {
					s_best = s.copy();
					score_best = score_cur_state;
					best_improve_iters += 1;
				}

				std::cout << "  [hard-reset] reinitialized after "
					<< hard_reset_after_blocks
					<< " non-improving blocks"
					<< "  seed=" << reset_seed
					<< "  score=(" << std::fixed << std::setprecision(10)
					<< score_cur_state.primary
					<< ", " << std::setprecision(4)
					<< score_cur_state.compactness << ")\n";
				continue;
			}

			if (hup < limit) {
				T = std::max(1e-12, cooling_alpha * T);
			} else {
				double progress = static_cast<double>(acc) / std::max(1.0, static_cast<double>(max_acc));
				if (use_time_budget) {
					progress = progress_time_ratio;
				}
				double beta = dynamic_beta(progress * 100.0, x_ini, x_end, y_ini, y_end);
				T = std::max(1e-12, T + beta);
				hup = 0;
			}
		} // end while acc < max_acc

		const auto run_end = std::chrono::steady_clock::now();
		double run_elapsed = std::chrono::duration<double>(run_end - run_start).count();

		run_results.push_back(score_best);
		std::cout << std::fixed << std::setprecision(10)
				  << "[run " << (run + 1) << "/" << runs << "]  "
				  << "best=(" << score_best.primary
				  << ", " << std::setprecision(4) << score_best.compactness << ")"
				  << "  evals=" << acc
				  << "  best-improve-iters=" << best_improve_iters
				  << "  elapsed=" << std::setprecision(3) << run_elapsed << "s\n";

		std::cout << std::setprecision(4) << "  LLH (idx  use  improve-best  improve-inc  acc-worse  val):\n";
		for (int i = 0; i < N; ++i) {
			std::cout << "    " << std::setw(2) << i
					  << "  " << std::setw(6) << usage[i]
					  << "  " << std::setw(12) << best_improve_cnt[i]
					  << "  " << std::setw(11) << improved_cnt[i]
					  << "  " << std::setw(9) << accepted_worse[i]
					  << "  " << std::setprecision(4) << val[i] << "\n";
		}

		if (!run_best_per_second.empty()) {
			std::cout << std::fixed << std::setprecision(10)
					  << "  Best lexicographic score per second (init=("
					  << score_init.primary << ", "
					  << std::setprecision(4) << score_init.compactness
					  << ")):\n";

			for (int sec = 0; sec <= static_cast<int>(run_elapsed); ++sec) {
				auto it = run_best_per_second.find(sec);
				if (it == run_best_per_second.end()) continue;

				const LexScore best_at_sec = it->second;
				const double improvement =
					score_init.primary - best_at_sec.primary;
				const double pct = (score_init.primary > 0.0)
					? 100.0 * improvement / score_init.primary
					: 0.0;

				std::cout << "    t=" << std::setw(3) << sec
						  << "s  best=(" << std::setprecision(10)
						  << best_at_sec.primary << ", "
						  << std::setprecision(4)
						  << best_at_sec.compactness << ")"
						  << "  primary_improvement="
						  << std::setprecision(10) << improvement
						  << " / " << std::setprecision(2)
						  << pct << "%\n";
			}
		}

		if (!have_overall ||
			lex_better(score_best, overall_best_score)) {
			overall_best = s_best.copy();
			overall_best_score = score_best;
			have_overall = true;
		}
	}

	const auto wall_end = std::chrono::steady_clock::now();
	double wall_elapsed = std::chrono::duration<double>(wall_end - wall_start).count();

	// Summary statistics across all runs.
	if (runs > 1) {
		LexScore best_run = run_results.front();
		LexScore worst_run = run_results.front();

		double primary_sum = 0.0;
		double primary_sum2 = 0.0;

		for (const LexScore& score : run_results) {
			if (lex_better(score, best_run)) {
				best_run = score;
			}
			if (lex_better(worst_run, score)) {
				worst_run = score;
			}

			primary_sum += score.primary;
			primary_sum2 += score.primary * score.primary;
		}

		const double primary_mean =
			primary_sum / static_cast<double>(runs);
		const double primary_var =
			primary_sum2 / static_cast<double>(runs)
			- primary_mean * primary_mean;
		const double primary_sd =
			primary_var > 0.0 ? std::sqrt(primary_var) : 0.0;

		std::vector<double> best_primary_compactness;
		for (const LexScore& score : run_results) {
			LexScore same_primary_reference{
				best_run.primary,
				score.compactness
			};
			if (lex_primary_equal(score, same_primary_reference)) {
				best_primary_compactness.push_back(score.compactness);
			}
		}

		double compactness_min = 0.0;
		double compactness_max = 0.0;
		double compactness_mean = 0.0;
		if (!best_primary_compactness.empty()) {
			compactness_min = *std::min_element(
				best_primary_compactness.begin(),
				best_primary_compactness.end());
			compactness_max = *std::max_element(
				best_primary_compactness.begin(),
				best_primary_compactness.end());
			for (double value : best_primary_compactness) {
				compactness_mean += value;
			}
			compactness_mean /=
				static_cast<double>(best_primary_compactness.size());
		}

		std::cout << std::fixed << std::setprecision(10)
				  << "\n=== Summary over " << runs << " runs ==="
				  << "\n  lex-best  = (" << best_run.primary
				  << ", " << std::setprecision(4)
				  << best_run.compactness << ")"
				  << "\n  lex-worst = (" << std::setprecision(10)
				  << worst_run.primary << ", "
				  << std::setprecision(4)
				  << worst_run.compactness << ")"
				  << "\n  primary mean    = "
				  << std::setprecision(10) << primary_mean
				  << "\n  primary std-dev = " << primary_sd
				  << "\n  compactness among best-primary runs"
				  << " (n=" << best_primary_compactness.size() << ")"
				  << ": min=" << std::setprecision(4) << compactness_min
				  << " mean=" << compactness_mean
				  << " max=" << compactness_max << "\n";
	}

	std::cout << std::setprecision(3)
			  << "Total wall time: " << wall_elapsed << "s\n";

	write_solution(overall_best, output_path);
	return 0;
}

// Run: g++ -std=c++17 -O3 -march=native -I/opt/conda/include -o wta_solver_hhasa_rl main_hhasa_rl.cpp && ./wta_solver_hhasa_rl data/scenario_022.json --runs 1 --search-seconds 5 --macc 100000 --iiter 100 --seed 42 --rl 3 --hard-reset-blocks 6
// python check_solution.py data/scenario_022.json data/scenario_022_solution.json && python plot.py data/scenario_022.json data/scenario_022_solution.json --out plot.png