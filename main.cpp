// main.cpp — WTA GRASP solver (C++)
// Requires nlohmann/json:  sudo apt install nlohmann-json3-dev
//
// Build:  g++ -std=c++17 -O2 -o wta main.cpp
// Run:    ./wta [scenario.json] [--restarts N] [--search-seconds S] [--terminate-by restarts|search-seconds] [--alpha A] [--seed S]

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
        wi.id                  = item["ID"];
        wi.code                = item["Code"];
        wi.type                = item["Type"];
        wi.min_range           = item["MinRange"];
        wi.max_range           = item["MaxRange"];
        wi.min_altitude        = item["MinAltitude"];
        wi.max_altitude        = item["MaxAltitude"];
        wi.azimuth_from_deg    = item["AzimuthFromDeg"];
        wi.azimuth_to_deg      = item["AzimuthToDeg"];
        wi.elevation_min_deg   = item["ElevationMinDeg"];
        wi.elevation_max_deg   = item["ElevationMaxDeg"];
        wi.max_shots_per_target = item["MaxShotsPerTarget"];
        wi.rounds_per_burst    = item["RoundsPerBurst"];
        wi.burst_interval      = item["BurstInterval"];
        wi.reload_time         = item["ReloadTime"];
        winfo_map[wi.code]     = wi;
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

        const WeaponInfo& wi    = winfo_map.at(w.info_code);
        sc.burst_dur[w.id]      = wi.burst_duration();
        sc.max_shots[w.id]      = wi.max_shots_per_target;
        sc.vessel_id_map[w.id]  = w.vessel_id;
        weapon_info_code[w.id]  = w.info_code;
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
            {"AmmoUsed",   a.ammo_used},
            {"PKill",      a.pkill},
            {"FireTime",   a.fire_time},
            {"EndTime",    a.end_time}
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
// print_solution
// ---------------------------------------------------------------------------
static void print_solution(const Solution& sol, int rank) {
    auto assignments = sol.assignments();
    std::sort(assignments.begin(), assignments.end(),
        [](const Assignment& a, const Assignment& b) {
            if (a.vessel_id != b.vessel_id) return a.vessel_id < b.vessel_id;
            if (a.weapon_id != b.weapon_id) return a.weapon_id < b.weapon_id;
            return a.target_id < b.target_id;
        });

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Solution #" << rank
              << "  objective=" << std::fixed << std::setprecision(6) << sol.objective()
              << "  assignments=" << assignments.size() << "\n";
    std::cout << std::string(60, '=') << "\n";
    for (auto& a : assignments) {
        std::cout << "  vessel=" << a.vessel_id
                  << "  weapon=" << std::setw(4) << a.weapon_id
                  << "  target=" << std::setw(4) << a.target_id
                  << "  bursts=" << a.ammo_used
                  << "  pkill=" << std::setprecision(4) << a.pkill
                  << "  fire=" << std::setprecision(2) << a.fire_time << "s"
                  << "  end="  << a.end_time << "s\n";
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    enum class TerminationMode { Restarts, SearchSeconds };

    std::string scenario_path = "/workspaces/WTA/data/scenario_001.json";
    std::string output_path;
    int    restarts            = 5;    // number of restarts or minimum restarts when using search-time termination
    double alpha               = 0.85; // GRASP alpha parameter: 0.0 = pure random, 1.0 = greedy    
    uint32_t seed              = 42;
    double search_seconds      = 5.0;
    TerminationMode term_mode  = TerminationMode::Restarts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--restarts"       && i + 1 < argc) restarts       = std::stoi(argv[++i]);
        else if (arg == "--alpha"     && i + 1 < argc) alpha          = std::stod(argv[++i]);
        else if (arg == "--seed"      && i + 1 < argc) seed           = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--search-seconds" && i + 1 < argc) search_seconds = std::stod(argv[++i]);
        else if (arg == "--terminate-by" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "restarts") term_mode = TerminationMode::Restarts;
            else if (mode == "search-seconds") term_mode = TerminationMode::SearchSeconds;
            else throw std::runtime_error("unknown termination mode: " + mode);
        }
        else if (arg == "--output"    && i + 1 < argc) output_path    = argv[++i];
        else if (arg[0] != '-')                        scenario_path  = arg;
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

    std::cout << "\nRunning GRASP: term_mode="
              << (term_mode == TerminationMode::Restarts ? "restarts" : "search-seconds")
              << "  restarts=" << restarts
              << "  search_seconds=" << search_seconds
              << "  alpha=" << alpha << "  seed=" << seed << "\n";

    std::mt19937 rng(seed);
    std::vector<Solution> solutions;

    auto t0 = std::chrono::steady_clock::now();
    auto elapsed_s = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    int r = 0;
    while ((term_mode == TerminationMode::Restarts && r < restarts)
        || (term_mode == TerminationMode::SearchSeconds && (r < restarts || elapsed_s() < search_seconds))) {
        Solution sol = Solution::empty(
            sc.weapons, sc.targets, sc.p_ij, sc.windows,
            sc.burst_dur, sc.max_shots, sc.vessel_id_map, sc.horizon);
        grasp_construction(sol, alpha, rng);
        double obj = sol.objective();
        ++r;
        std::cout << "  restart " << std::setw(4) << r
                  << "  obj=" << std::fixed << std::setprecision(6) << obj << "\n";
        solutions.push_back(std::move(sol));
    }
    double elapsed = elapsed_s();
    std::cout << "\nCompleted in " << std::fixed << std::setprecision(3) << elapsed
              << "s  (" << r << " restarts)\n";

    std::sort(solutions.begin(), solutions.end(),
        [](const Solution& a, const Solution& b){ return a.objective() < b.objective(); });

    /* for (int rank = 1; rank <= static_cast<int>(solutions.size()); ++rank)
        print_solution(solutions[rank - 1], rank); */

    std::cout << "\nBest objective: " << std::fixed << std::setprecision(6)
              << solutions[0].objective() << "\n";

    write_solution(solutions[0], output_path);
    return 0;
}

// Run: g++ -std=c++17 -O3 -march=native -I/opt/conda/include -o wta_solver main.cpp && ./wta_solver data/scenario_050.json [scenario.json] [--restarts N] [--search-seconds S] [--terminate-by restarts|search-seconds] [--alpha A] [--seed S]
// Check && plot: python check_solution.py data/scenario_022.json /workspaces/kudzu/data/scenario_022_solution.json && python plot.py data/scenario_022.json /workspaces/kudzu/data/scenario_022_solution.json --out plot.png