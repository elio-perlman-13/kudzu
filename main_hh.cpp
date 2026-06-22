// main_hh.cpp — WTA baseline + Lean-style perturbative HH (single path)
// Requires nlohmann/json:  sudo apt install nlohmann-json3-dev
//
// Build:  g++ -std=c++17 -O3 -march=native -I/opt/conda/include -o wta_solver_hh main_hh.cpp
// Run:    ./wta_solver_hh [scenario.json] [--restarts N] [--alpha A] [--seed S] [--search-seconds T]

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

#include "json.hpp"
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

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
	std::string scenario_path = "/workspaces/WTA/data/scenario_001.json";
	std::string output_path;
	std::string config_path   = "config.json";
	int    restarts = 1;
	double alpha    = 0.85;
	uint32_t seed   = 42;
	double search_seconds = 5.0;

	// --- load config.json ("lean_gihh" section) before parsing CLI args ---
	{
		std::ifstream cfg_f(config_path);
		if (cfg_f) {
			try {
				json cfg = json::parse(cfg_f);
				if (cfg.contains("lean_gihh")) {
					auto& g = cfg["lean_gihh"];
					if (g.contains("seed"))           seed           = g["seed"].get<uint32_t>();
					if (g.contains("restarts"))       restarts       = g["restarts"].get<int>();
					if (g.contains("alpha"))          alpha          = g["alpha"].get<double>();
					if (g.contains("search_seconds")) search_seconds = g["search_seconds"].get<double>();
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
		else if (arg == "--alpha"  && i + 1 < argc) alpha = std::stod(argv[++i]);
		else if (arg == "--seed"   && i + 1 < argc) seed  = static_cast<uint32_t>(std::stoul(argv[++i]));
		else if (arg == "--search-seconds" && i + 1 < argc) search_seconds = std::stod(argv[++i]);
		else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
		else if (arg[0] != '-') scenario_path = arg;
	}

	// default output path: replace .json suffix with _solution.json
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
			  << "  pairs="   << sc.windows.size()
			  << "  horizon=" << sc.horizon << "s\n";

	std::cout << "\nRunning baseline init + perturbative HH"
			  << "  init_restarts=" << restarts
			  << "  alpha=" << alpha
			  << "  seed=" << seed
			  << "  search_seconds=" << search_seconds << "\n";

	std::mt19937 rng(seed);
	Solution incumbent = grasp(
		sc.weapons, sc.targets, sc.p_ij, sc.windows,
		sc.burst_dur, sc.max_shots, sc.vessel_id_map,
		sc.horizon, alpha, restarts, seed);

	Solution runbest = incumbent.copy();
	Solution best    = incumbent.copy();

	double f_initial   = incumbent.objective();
	double f_incumbent = f_initial;
	double f_runbest   = f_initial;
	double f_best      = f_initial;

	constexpr int bestlist_size = 6;
	constexpr int K = 100;

	std::vector<double> runbest_list(bestlist_size, f_initial);
	std::vector<double> best_list(bestlist_size, f_initial);
	int index = 1;
	int counter_K = 0;
	bool stuck = false;
	bool restart_disabled = false;

	constexpr int N = perturbation::kLLHCount;
	std::vector<double> t_spent_ms(N, 0.0);
	std::vector<int> n_best(N, 0);
	std::vector<int> n_improve(N, 0);
	std::vector<int> n_worsen(N, 0);
	std::vector<int> usage(N, 0);
	std::vector<double> val(N, 0.2);
	std::vector<double> pr(N, 1.0 / static_cast<double>(N));
	int n_best_total = 0;

	int perturbations_attempted = 0;
	int perturbations_changed = 0;

	std::vector<std::string> llh_names = {
		"M2_RUIN_RANDOM__H_SURV_THREAT_TIE",
		"M4_RUIN_OVER_COVERED__H_EXCLUSIVE_RESERVE",
		"M6_RUIN_CONGESTED__H_WINDOW_CLOSURE",
		"M7_RUIN_LOW_MARGINAL__H_COVER_FIRST",
		"M8_RUIN_CONGESTED__H_BACKLOG_RELIEF",
		"M10_RUIN_RANDOM__H_SPREAD_THEN_FOCUS",
		"M11_RUIN_CONGESTED__H_ANTI_BOTTLENECK",
		"M12_RUIN_OVER_COVERED__H_OPPORTUNITY_LOCK",
		"M13_RUIN_LOW_MARGINAL__H_MARGINAL_OBJECTIVE_DROP",
		"M14_RUIN_OVER_COVERED__H_KILL_CHAIN_FINISHER",
		"M15_RUIN_CONGESTED__H_FUTURE_FLEX_PRESERVER",
		"M16_RUIN_RANDOM__H_BASELINE_TWO_STAGE",
		"L1_REASSIGN_BEST_DELTA",
		"L2_SWAP_BEST_PAIR"
	};

	constexpr int N_MACROS = N - 1; // 13: indices 0..12

	auto roulette_macro = [&]() -> int {
		double norm = 0.0;
		for (int i = 0; i < N_MACROS; ++i) norm += pr[i];
		if (norm <= 0.0) return std::uniform_int_distribution<int>(0, N_MACROS - 1)(rng);
		double pivot = std::uniform_real_distribution<double>(0.0, norm)(rng);
		double accum = 0.0;
		for (int i = 0; i < N_MACROS; ++i) {
			accum += pr[i];
			if (accum >= pivot) return i;
		}
		return N_MACROS - 1;
	};

	constexpr double diff[4] = {0.0025, 0.00025, -0.00025, -0.00005};

	auto init_from_baseline = [&](uint32_t local_seed) {
		Solution s = grasp(
			sc.weapons, sc.targets, sc.p_ij, sc.windows,
			sc.burst_dur, sc.max_shots, sc.vessel_id_map,
			sc.horizon, alpha, restarts, local_seed);
		incumbent = s.copy();
		runbest = s.copy();
		f_incumbent = f_runbest = s.objective();
		runbest_list.assign(bestlist_size, f_incumbent);
		counter_K = 0;
		index = 1;
		stuck = false;
		std::cout << "Stucked, Initialized from baseline with seed=" << local_seed
			 << " objective=" << std::fixed << std::setprecision(10) << f_incumbent << "\n";
	};

	auto begin = std::chrono::steady_clock::now();
	std::unordered_map<int, double> run_best_per_second;
	run_best_per_second[0] = f_initial;
	auto elapsed_seconds = [&]() {
		auto now = std::chrono::steady_clock::now();
		return std::chrono::duration<double>(now - begin).count();
	};

	while (elapsed_seconds() < search_seconds) {
		double frac = std::min(1.0, elapsed_seconds() / std::max(1e-9, search_seconds));

		// Restart logic (Lean-style): active in first half only.
		if (!restart_disabled) {
			if (frac <= 0.5) {
				if (stuck) {
					if (f_runbest < f_best) {
						best = runbest.copy();
						f_best = f_runbest;
						best_list = runbest_list;
					}
					init_from_baseline(seed + static_cast<uint32_t>(perturbations_attempted + 1));
				}
			} else {
				restart_disabled = true;
				stuck = false;
				if (f_best < f_runbest) {
					incumbent = best.copy();
					runbest = best.copy();
					f_incumbent = f_runbest = f_best;
					runbest_list = best_list;
					counter_K = 0;
					index = 1;
				}
			}
		}

		// Update pr[] using n_best/t_spent for macros only.
		double tf = 1.0 - frac;
		double expo = 1.0 + 3.0 * tf * tf * tf;
		for (int i = 0; i < N_MACROS; ++i) {
			double denom = std::max(1.0, t_spent_ms[i]);
			pr[i] = std::pow((static_cast<double>(n_best[i]) + 1.0) / denom, expo);
		}

		int heur = roulette_macro();
		usage[heur] += 1;

		Solution proposed = incumbent.copy();
		auto app_t0 = std::chrono::steady_clock::now();
		bool changed = false;
		try {
			changed = perturbation::apply_llh(
				static_cast<perturbation::LLHId>(heur),
				proposed, rng,
				perturbation::ApplyParams{val[heur]});
		} catch (const std::exception& ex) {
			std::cerr << "LLH threw std::exception: idx=" << heur
					  << " name=" << llh_names[heur]
					  << " what=" << ex.what() << "\n";
			throw;
		} catch (...) {
			std::cerr << "LLH threw unknown exception: idx=" << heur
					  << " name=" << llh_names[heur] << "\n";
			throw;
		}
		t_spent_ms[heur] += std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - app_t0).count();
		perturbations_attempted += 1;
		if (changed) perturbations_changed += 1;

		double f_proposed = proposed.objective();

		int x = 3; // 0:new_best 1:improve 2:worsen 3:equal
		if (f_proposed < f_incumbent) {
			x = 1;
			n_improve[heur] += 1;
			if (f_proposed < f_runbest) {
				runbest = proposed.copy();
				f_runbest = f_proposed;
				n_best[heur] += 1;
				n_best_total += 1;
				x = 0;
			}
		} else if (f_proposed > f_incumbent) {
			x = 2;
			n_worsen[heur] += 1;
		}
		val[heur] += diff[x];
		val[heur] = std::clamp(val[heur], 0.2, 1.0);

		// AILLA-like acceptance.
		bool accept = false;
		if (f_proposed < f_incumbent) {
			if (f_proposed < runbest_list[0]) {
				stuck = false;
				runbest_list.insert(runbest_list.begin(), f_proposed);
				if (static_cast<int>(runbest_list.size()) > bestlist_size) runbest_list.pop_back();
				index = 1;
				counter_K = 0;
			}
			accept = true;
		} else if (std::fabs(f_proposed - f_incumbent) <= 1e-12) {
			accept = true;
		} else {
			index = std::clamp(index, 0, bestlist_size - 1);
			accept = (f_proposed <= runbest_list[index]);
			counter_K += 1;
			if (counter_K >= K) {
				counter_K = 0;
				index += 1;
				if (index >= bestlist_size) {
					stuck = true;
					index = bestlist_size - 1;
				}
			}
		}

		if (accept) {
			incumbent = proposed.copy();
			f_incumbent = f_proposed;
		}

		// Record cumulative best objective so far at this second.
		{
			double elapsed = elapsed_seconds();
			int sec = static_cast<int>(elapsed);
			double best_so_far = std::min(f_best, f_runbest);
			auto it = run_best_per_second.find(sec);
			if (it == run_best_per_second.end()) {
				run_best_per_second[sec] = best_so_far;
			} else {
				it->second = std::min(it->second, best_so_far);
			}
		}

	}

	double run_elapsed = elapsed_seconds();

	if (f_runbest < f_best) {
		best = runbest.copy();
		f_best = f_runbest;
		best_list = runbest_list;
	}

	if (!run_best_per_second.empty()) {
		std::cout << std::fixed << std::setprecision(10)
				  << "  Best objective per second (init=" << f_initial << "):\n";
		for (int sec = 0; sec <= static_cast<int>(run_elapsed); ++sec) {
			if (run_best_per_second.find(sec) != run_best_per_second.end()) {
				double best_at_sec = run_best_per_second[sec];
				double improvement = f_initial - best_at_sec;
				double pct = (f_initial > 0.0) ? 100.0 * improvement / f_initial : 0.0;
				std::cout << "    t=" << std::setw(3) << sec << "s  best=" 
						  << best_at_sec 
						  << "  (improvement=" << std::setprecision(10) << improvement 
						  << " / " << std::setprecision(2) << pct << "%)\n";
			}
		}
	}

	std::cout << "\nObjective before perturbation: " << std::fixed << std::setprecision(10) << f_initial << "\n";
	std::cout << "Objective after perturbation:  " << std::fixed << std::setprecision(10) << f_best << "\n";
	std::cout << "Perturbations attempted: " << perturbations_attempted
			  << "  changed: " << perturbations_changed << "\n";

	std::cout << "\nLLH usage/value summary:\n";
	std::cout << "idx  uses  n_improve  n_worsen  n_best   t_spent_ms   val      score\n";
	for (int i = 0; i < N; ++i) {
		double score = (static_cast<double>(n_best[i]) + 1.0) / std::max(1.0, t_spent_ms[i]);
		std::cout << std::setw(2) << i
				  << "  " << std::setw(5) << usage[i]
				  << "  " << std::setw(9) << n_improve[i]
				  << "  " << std::setw(8) << n_worsen[i]
				  << "  " << std::setw(6) << n_best[i]
				  << "  " << std::setw(11) << std::fixed << std::setprecision(3) << t_spent_ms[i]
				  << "  " << std::setw(6) << std::setprecision(4) << val[i]
				  << "  " << std::setw(9) << std::setprecision(6) << score
				  << "  " << llh_names[i] << "\n";
	}

	write_solution(best, output_path);
	return 0;
}

// Run:
// g++ -std=c++17 -O3 -march=native -I/opt/conda/include -o wta_solver_hh main_hh.cpp && ./wta_solver_hh data/scenario_035.json
// Check && plot: python check_solution.py data/scenario_035.json /workspaces/WTA/data/scenario_035_solution.json && python plot.py data/scenario_035.json /workspaces/WTA/data/scenario_035_solution.json --out plot.png

//python check_solution.py data/scenario_ak630_5uav_1yj83.json data/scenario_ak630_5uav_1yj83_solution.json && python plot.py data/scenario_ak630_5uav_1yj83.json data/scenario_ak630_5uav_1yj83_solution.json --out plot.png
