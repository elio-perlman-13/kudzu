#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

#include "heuristic.hpp"
#include "perturbation.hpp"

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Scenario — owns all static tables; Solution holds const pointers into them
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// load_scenario — parse JSON into a Scenario
// ---------------------------------------------------------------------------
static Scenario load_scenario(const std::string& path) {
	std::ifstream f(path);
	if (!f) throw std::runtime_error("cannot open: " + path);
	json data = json::parse(f);

	Scenario sc;

	// --- weapon infos ---
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

	// --- target infos ---
	std::unordered_map<std::string, TargetInfo> tinfo_map;
	for (auto& item : data["target_infos"]) {
		TargetInfo ti;
		ti.id          = item["ID"];
		ti.code        = item["Code"];
		ti.description = item.value("Description", "");
		ti.type        = item["Type"];
		tinfo_map[ti.code] = ti;
	}

	// --- probability table ---
	// key: "weapon_code|target_code"
	std::unordered_map<std::string, double> prob_map;
	for (auto& row : data["probability_table"]) {
		std::string key = std::string(row["WTAWeaponInfoCode"]) + "|"
						+ std::string(row["WTATargetInfoCode"]);
		prob_map[key] = row["Score"];
	}

	auto& req = data["assignment_request"];

	// --- weapons + static tables ---
	std::unordered_map<int, std::string> weapon_info_code; // wid -> info_code
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

	// --- targets ---
	std::unordered_map<int, std::string> target_info_code; // tid -> info_code
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

	// --- engagement windows + p_ij ---
	for (auto& [key_str, ab] : data["engagement_windows"].items()) {
		// key_str = "wid_tid"
		auto sep = key_str.find('_');
		int wid = std::stoi(key_str.substr(0, sep));
		int tid = std::stoi(key_str.substr(sep + 1));

		double a = ab[0], b = ab[1];

		std::string pkey = weapon_info_code.at(wid) + "|" + target_info_code.at(tid);
		auto pit = prob_map.find(pkey);
		if (pit == prob_map.end() || pit->second <= 0.0) continue;

		uint64_t k = pair_key(wid, tid);
		sc.windows[k] = {a, b};
		sc.p_ij[k]    = pit->second;
	}

	if (!sc.windows.empty()) {
		sc.horizon = 0.0;
		for (auto& [k, ab] : sc.windows)
			sc.horizon = std::max(sc.horizon, ab.second);
	}

	return sc;
}

// ---------------------------------------------------------------------------
// write_solution — emit best solution as JSON (WTAAssignmentResponse schema)
// ---------------------------------------------------------------------------
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
	if (std::fabs(x_end - x_ini) < 1e-12) return y_ini;
	double m = (y_end - y_ini) / (x_end - x_ini);
	return m * progress_pct + y_ini;
}

static int roulette_index(const std::vector<double>& weights, std::mt19937& rng) {
	double sum = 0.0;
	for (double w : weights) sum += std::max(0.0, w);
	if (sum <= 0.0) {
		return std::uniform_int_distribution<int>(0, static_cast<int>(weights.size()) - 1)(rng);
	}
	double pivot = std::uniform_real_distribution<double>(0.0, sum)(rng);
	double acc = 0.0;
	for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
		acc += std::max(0.0, weights[i]);
		if (acc >= pivot) return i;
	}
	return static_cast<int>(weights.size()) - 1;
}

// UCB1 — exact EVRPSARL.m case 3: Suc/Selected + sqrt(2*ln(k)/Selected)
// suc = success counts (delta<=0), selected = pull counts
static int select_llh_ucb(
	const std::vector<double>& suc,
	const std::vector<int>& selected,
	int k,
	std::mt19937& /*rng*/)
{
	const int n = static_cast<int>(suc.size());
	// Try each arm at least once in order (mirrors MATLAB: action=k for k<=nbandits)
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

// Thompson Sampling — exact EVRPSARL.m case 2: betarand(Suc+1, Fail+1) with Fail=0
// = sample from Beta(Suc+1, 1) via Gamma ratio
static int select_llh_thompson(
	const std::vector<double>& suc,
	std::mt19937& rng)
{
	const int n = static_cast<int>(suc.size());
	double best = -std::numeric_limits<double>::infinity();
	int argbest = 0;
	std::uniform_real_distribution<double> u01(std::numeric_limits<double>::min(), 1.0);
	for (int i = 0; i < n; ++i) {
		// Beta(a,1) with a=suc+1 has inverse-CDF sample: U^(1/a), U~Unif(0,1).
		// This avoids per-arm Gamma/Exponential sampling overhead.
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

// bb_threshold: trigger only if rand > bb (paper Rutas.m: if rand()>bb).
// bb = 1.0 - acc/(max_acc*9), ranges from 1.0 (never) -> ~0.89 (11% chance at end).
static bool adjust_station_block(
	Solution& s,
	std::mt19937& rng,
	double p_relocate,
	double p_eliminate,
	double bb_threshold,
	const std::vector<double>& val)
{
	if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) <= bb_threshold) return false;

	std::vector<double> action_weights = {
		std::max(0.0, p_relocate),
		std::max(0.0, p_eliminate)
	};
	int action = roulette_index(action_weights, rng);

	if (action == 0) {
		// Relocate-like action: focused reassignment/swap local moves.
		if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < 0.6) {
			return perturbation::apply_llh(
				perturbation::LLHId::L1_REASSIGN_BEST_DELTA,
				s,
				rng,
				perturbation::ApplyParams{1.0});
		}
		return perturbation::apply_llh(
			perturbation::LLHId::L2_SWAP_BEST_PAIR,
			s,
			rng,
			perturbation::ApplyParams{1.0});
	}

	// Eliminate-like action: stronger destroy/rebuild macro move.
	double v = val.empty() ? 0.8 : std::clamp(val[0] + 0.25, 0.2, 1.0);
	return perturbation::apply_llh(
		perturbation::LLHId::M2_RUIN_RANDOM__H_SURV_THREAT_TIE,
		s,
		rng,
		perturbation::ApplyParams{v});
}

int main(int argc, char* argv[]) {
	std::string scenario_path = "/workspaces/WTA/data/scenario_001.json";
	std::string output_path;

	int restarts = 1;
	double grasp_alpha = 0.85;
	uint32_t seed = 42;

	int    max_acc       = 0;     // 0 = auto: 25000 * nc
	int    iiter         = 0;     // 0 = auto: 40 * nc
	int    runs          = 1;
	double search_seconds = 0.0;  // 0 = disabled (evaluation-budget mode)
	double temp_init     = -1.0;
	double cooling_alpha = 0.99;  // paper
	int    limit         = 20;    // paper
	double x_ini         = 0.0;   // paper
	double x_end         = 90.0;  // paper
	double y_ini         = 1.0;   // paper
	double y_end         = 0.05;  // paper
	double p_relocate    = 0.60;  // paper
	double p_eliminate   = 0.40;  // paper
	int    rl_type       = 2;     // paper default: 2=Thompson Sampling (0=Rand,1=eGreedy,3=UCB1)
	bool   fast_mode     = false; // disable expensive local LLHs (12,13)
	bool   strict_bandit_success = true; // count success only on strict objective improvements
	bool   auto_prune_dead_llh = true;   // disable LLHs with sustained zero improvements
	int    prune_min_usage = 300;        // minimum uses before dead-arm pruning
	double explore_eps = 0.03;           // epsilon exploration around policy selector
	int    kick_blocks = 6;              // stagnation blocks before diversification kick
	double kick_val = 0.9;               // ruin intensity for diversification kick

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--restarts" && i + 1 < argc) restarts = std::stoi(argv[++i]);
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
		else if (arg == "--pr" && i + 1 < argc) p_relocate = std::stod(argv[++i]);
		else if (arg == "--pe"   && i + 1 < argc) p_eliminate = std::stod(argv[++i]);
		else if (arg == "--runs" && i + 1 < argc) runs        = std::stoi(argv[++i]);
		else if (arg == "--search-seconds" && i + 1 < argc) search_seconds = std::stod(argv[++i]);
		else if (arg == "--rl"   && i + 1 < argc) rl_type     = std::stoi(argv[++i]);
		else if (arg == "--fast-mode") fast_mode = true;
		else if (arg == "--non-strict-success") strict_bandit_success = false;
		else if (arg == "--no-auto-prune") auto_prune_dead_llh = false;
		else if (arg == "--prune-min-usage" && i + 1 < argc) prune_min_usage = std::stoi(argv[++i]);
		else if (arg == "--eps" && i + 1 < argc) explore_eps = std::stod(argv[++i]);
		else if (arg == "--kick-blocks" && i + 1 < argc) kick_blocks = std::stoi(argv[++i]);
		else if (arg == "--kick-val" && i + 1 < argc) kick_val = std::stod(argv[++i]);
		else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
		else if (arg[0] != '-') scenario_path = arg;
	}

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

	// Derive nc-scaled parameters (paper: MAcc = 25000*nc, IIter = 40*nc)
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
			  << "  rl=" << rl_type
			  << "  runs=" << runs << "\n";

	constexpr int N = perturbation::kLLHCount;

	// Track overall best across all runs.
	Solution overall_best;
	double   overall_best_fit = std::numeric_limits<double>::infinity();
	bool     have_overall     = false;

	std::vector<double> run_results;
	run_results.reserve(runs);

	// Time-based tracking: second -> best objective found by that second
	std::unordered_map<int, double> best_per_second;

	const auto wall_start = std::chrono::steady_clock::now();

	std::vector<char> llh_enabled(N, 1);
	if (fast_mode && N >= 14) {
		llh_enabled[12] = 0;
		llh_enabled[13] = 0;
	}
	int enabled_count = 0;
	for (char e : llh_enabled) enabled_count += (e ? 1 : 0);
	std::cout << "  fast_mode=" << (fast_mode ? 1 : 0)
			  << "  active_llh=" << enabled_count << "/" << N << "\n";
	std::cout << "  strict_success=" << (strict_bandit_success ? 1 : 0)
			  << "  auto_prune=" << (auto_prune_dead_llh ? 1 : 0)
			  << "  eps=" << explore_eps
			  << "  kick_blocks=" << kick_blocks
			  << "  kick_val=" << kick_val << "\n";

	for (int run = 0; run < runs; ++run) {
		uint32_t run_seed = seed + static_cast<uint32_t>(run);
		std::mt19937 rng(run_seed);

		const auto run_start = std::chrono::steady_clock::now();

		// Per-run time tracking
		std::unordered_map<int, double> run_best_per_second;

		// --- Initialization: GRASP ---
		Solution s = grasp(
			sc.weapons, sc.targets, sc.p_ij, sc.windows,
			sc.burst_dur, sc.max_shots, sc.vessel_id_map,
			sc.horizon, grasp_alpha, restarts, run_seed);

		double fit_init = s.objective();
		std::cout << std::fixed << std::setprecision(10)
				  << "[run " << (run + 1) << "/" << runs
				  << "  seed=" << run_seed
				  << "]  GRASP init obj=" << fit_init << "\n";

		// Record initial solution at t=0
		run_best_per_second[0] = fit_init;

		Solution s_best   = s.copy();
		double   fit_best = fit_init;
		double   fit_cur_state = fit_init;

		double T0 = (temp_init > 0.0) ? temp_init
					                   : std::max(1e-6, 0.01 * std::max(1.0, fit_init));
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
		int stagnation_blocks = 0;
		const bool use_time_budget = (search_seconds > 0.0);

		while (acc < max_acc) {
			if (use_time_budget) {
				double elapsed = std::chrono::duration<double>(
					std::chrono::steady_clock::now() - run_start).count();
				if (elapsed >= search_seconds) break;
			}
		// Per-block bandit state (reset after each inner loop, matching EVRPSARL.m)
		std::vector<double> suc(N, 0.0);    // success count: delta<=0
		std::vector<int>    pulls(N, 0);    // selection count
		int total_pulls = 0;
		std::vector<char> llh_enabled_block = llh_enabled;
		if (auto_prune_dead_llh) {
			for (int i = 0; i < N; ++i) {
				if (!llh_enabled_block[i]) continue;
				if (usage[i] >= prune_min_usage && improved_cnt[i] == 0 && best_improve_cnt[i] == 0) {
					llh_enabled_block[i] = 0;
				}
			}
			int active_block = 0;
			for (char e : llh_enabled_block) active_block += (e ? 1 : 0);
			if (active_block == 0) llh_enabled_block = llh_enabled;
		}

		Solution block_best;
		bool block_best_updated = false;
		double fit_block_best = fit_cur_state;
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
			if (unit01(rng) < std::clamp(explore_eps, 0.0, 1.0)) {
				heur = select_random_enabled(llh_enabled_block, rng);
			} else if (rl_type == 3) {
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
				// Do not count as success; still count as selection (mirrors MATLAB Selected++)
				pulls[heur]++;
				total_pulls++;
				continue;
			}

			// Repair block: if the generated move breaks basic feasibility, try targeted local repairs.
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

			// Adjust Station proxy: bb varies from 1.0 (never) to ~0.89 (11% chance)
			// matching EVRPSARL.m Rutas(): bb = 1 - it/(MaxIter*9)
			double progress = static_cast<double>(acc) / std::max(1.0, static_cast<double>(max_acc));
			if (use_time_budget) {
				progress = progress_time_ratio;
			}
			double bb = 1.0 - progress / 9.0;
			adjust_station_block(cand, rng, p_relocate, p_eliminate, bb, val);

			double fit_cand = cand.objective();
			double delta = fit_cand - fit_cur_state;
			acc++;

			// Record cumulative best objective so far at this second
			{
				double elapsed = std::chrono::duration<double>(
					std::chrono::steady_clock::now() - run_start).count();
				int sec = static_cast<int>(elapsed);
				
				// Store the best found anywhere up to this point
				if (run_best_per_second.find(sec) == run_best_per_second.end()) {
					run_best_per_second[sec] = fit_best;
				} else {
					// Keep the better cumulative value
					run_best_per_second[sec] = std::min(run_best_per_second[sec], fit_best);
				}
			}

			bool accept = false;
			if (delta <= 0.0) {
				accept = true;
				if (strict_bandit_success) {
					if (delta < 0.0) suc[heur] += 1.0;
				} else {
					suc[heur] += 1.0;
				}
				if (delta < 0.0) improved_cnt[heur]++;
				if (delta < 0.0 && fit_cand < fit_best) best_improve_cnt[heur]++;
			} else {
				double p = std::exp(-delta / std::max(1e-12, T));
				double r = unit01(rng);
				if (r < p) {
					accept = true;
					accepted_worse[heur]++;
					// Note: Metropolis-accepted worsening moves are NOT rewarded in EVRPSARL.m
				}
			}

			pulls[heur]++;
			total_pulls++;

			if (accept) {
				s = std::move(cand);
				fit_cur_state = fit_cand;
				if (fit_cand < fit_block_best) {
					block_best = s.copy();
					fit_block_best = fit_cand;
					block_best_updated = true;
				}
			}

			int reward_case = 3; // 0: new best, 1: improve, 2: worsen, 3: equal
			if (delta < 0.0) reward_case = (fit_cand < fit_best) ? 0 : 1;
			else if (delta > 0.0) reward_case = 2;
			static constexpr double diff[4] = {0.0025, 0.00025, -0.00025, -0.00005};
			val[heur] = std::clamp(val[heur] + diff[reward_case], 0.2, 1.0);
		}

			if (block_best_updated && fit_block_best < fit_best) {
				s_best   = std::move(block_best);
				fit_best = fit_block_best;
				best_improve_iters += 1;
				hup = 0;
				stagnation_blocks = 0;
			} else {
				hup += 1;
				stagnation_blocks += 1;
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

			if (kick_blocks > 0 && stagnation_blocks >= kick_blocks) {
				Solution kicked = s_best.copy();
				perturbation::apply_llh(
					perturbation::LLHId::M10_RUIN_RANDOM__H_SPREAD_THEN_FOCUS,
					kicked,
					rng,
					perturbation::ApplyParams{std::clamp(kick_val, 0.2, 1.0)});
				s = std::move(kicked);
				fit_cur_state = s.objective();
				T = std::max(T, 0.5 * T0);
				stagnation_blocks = 0;
			}
		} // end while acc < max_acc

		const auto run_end = std::chrono::steady_clock::now();
		double run_elapsed = std::chrono::duration<double>(run_end - run_start).count();

		run_results.push_back(fit_best);
		std::cout << std::fixed << std::setprecision(10)
				  << "[run " << (run + 1) << "/" << runs << "]  "
				  << "best=" << fit_best
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

		// Print best objective per second
		if (!run_best_per_second.empty()) {
			std::cout << std::fixed << std::setprecision(10)
					  << "  Best objective per second (init=" << fit_init << "):\n";
			for (int sec = 0; sec <= static_cast<int>(run_elapsed); ++sec) {
				if (run_best_per_second.find(sec) != run_best_per_second.end()) {
					double best_at_sec = run_best_per_second[sec];
					double improvement = fit_init - best_at_sec;
					double pct = (fit_init > 0.0) ? 100.0 * improvement / fit_init : 0.0;
					std::cout << "    t=" << std::setw(3) << sec << "s  best=" 
							  << best_at_sec 
							  << "  (improvement=" << std::setprecision(10) << improvement 
							  << " / " << std::setprecision(2) << pct << "%)\n";
				}
			}
		}

		if (!have_overall || fit_best < overall_best_fit) {
			overall_best     = s_best.copy();
			overall_best_fit = fit_best;
			have_overall     = true;
		}
	} // end for run

	const auto wall_end = std::chrono::steady_clock::now();
	double wall_elapsed = std::chrono::duration<double>(wall_end - wall_start).count();

	// Summary statistics across all runs.
	if (runs > 1) {
		double sum = 0.0, sum2 = 0.0;
		double best_r = run_results[0], worst_r = run_results[0];
		for (double v : run_results) {
			sum  += v; sum2 += v * v;
			if (v < best_r)  best_r  = v;
			if (v > worst_r) worst_r = v;
		}
		double mean = sum / runs;
		double var  = sum2 / runs - mean * mean;
		double sd   = (var > 0.0) ? std::sqrt(var) : 0.0;
		std::cout << std::fixed << std::setprecision(10)
				  << "\n=== Summary over " << runs << " runs ==="
				  << "\n  best    = " << best_r
				  << "\n  worst   = " << worst_r
				  << "\n  mean    = " << mean
				  << "\n  std-dev = " << sd << "\n";
	}

	std::cout << std::setprecision(3)
			  << "Total wall time: " << wall_elapsed << "s\n";

	write_solution(overall_best, output_path);
	return 0;
}

// Run: g++ -std=c++17 -O3 -march=native -I/opt/conda/include -o wta_solver_hhasa_rl main_hhasa_rl.cpp && ./wta_solver_hhasa_rl data/scenario_022.json --runs 1 --search-seconds 5 --macc 100000 --iiter 100 --seed 42 --rl 3