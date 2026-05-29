#pragma once
#include <string>
#include <vector>
#include <optional>

// ---------------------------------------------------------------------------
// WeaponInfo — static weapon type properties
// ---------------------------------------------------------------------------
struct WeaponInfo {
    int    id;
    std::string code;
    int    type;
    double min_range;
    double max_range;
    double min_altitude;
    double max_altitude;
    double azimuth_from_deg;
    double azimuth_to_deg;
    double elevation_min_deg;
    double elevation_max_deg;
    int    max_shots_per_target;   // M_i cap per target
    int    rounds_per_burst;
    double burst_interval;         // duration of one burst
    double reload_time;            // gap after each burst

    double burst_duration() const { return burst_interval + reload_time; }
};

// ---------------------------------------------------------------------------
// TargetInfo — static target type properties
// ---------------------------------------------------------------------------
struct TargetInfo {
    int    id;
    std::string code;
    std::string description;
    int    type;
};

// ---------------------------------------------------------------------------
// Vessel — platform carrying weapons
// ---------------------------------------------------------------------------
struct Vessel {
    int    id;
    double x, y, z;
    double speed;
    double heading_x, heading_y, heading_z;
    double defense_radius;
};

// ---------------------------------------------------------------------------
// Weapon — a weapon instance mounted on a vessel
// ---------------------------------------------------------------------------
struct Weapon {
    int    id;
    int    vessel_id;
    int    ammo;
    std::string info_code;
    int    status;

    // resolved at load time
    const WeaponInfo* info = nullptr;

    double burst_duration()      const { return info->burst_duration(); }
    int    max_shots_per_target() const { return info->max_shots_per_target; }
};

// ---------------------------------------------------------------------------
// Target — a threat target to be engaged
// ---------------------------------------------------------------------------
struct Target {
    int    id;
    std::string info_code;
    double x, y, z;
    double vx, vy, vz;
    double speed;
    double threat_score;          // w_j — static threat weight

    // resolved at load time
    const TargetInfo* info = nullptr;
};
