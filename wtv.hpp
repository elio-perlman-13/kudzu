#pragma once
#include <string>
#include <vector>
#include <optional>

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
    int    max_shots_per_target;  
    int    rounds_per_burst;
    double burst_interval;       
    double reload_time;      

    double burst_duration() const { return burst_interval + reload_time; }
};

struct TargetInfo {
    int    id;
    std::string code;
    std::string description;
    int    type;
};

struct Vessel {
    int    id;
    double x, y, z;
    double speed;
    double heading_x, heading_y, heading_z;
    double defense_radius;
};

struct Weapon {
    int    id;
    int    vessel_id;
    int    ammo;
    std::string info_code;
    int    status;
    const WeaponInfo* info = nullptr;

    double burst_duration()      const { return info->burst_duration(); }
    int    max_shots_per_target() const { return info->max_shots_per_target; }
};

struct Target {
    int    id;
    std::string info_code;
    double x, y, z;
    double vx, vy, vz;
    double speed;
    double threat_score; 
    const TargetInfo* info = nullptr;
};
