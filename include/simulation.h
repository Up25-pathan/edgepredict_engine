#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <string>
#include "json.hpp"
#include <Eigen/Dense>

using json = nlohmann::json;

enum class NodeStatus {
    OK,
    FRACTURED
};

struct Node {
    Eigen::Vector3d position;
    double temperature = 20.0;
    double stress = 0.0;
    double strain = 0.0;
    double accumulated_wear = 0.0; // <-- NEW: To track Usui wear
    NodeStatus status = NodeStatus::OK;
};

struct Mesh {
    std::vector<Node> nodes;
};

class Simulation; // <-- NEW: Forward declaration

class FEASolver {
public:
    FEASolver(const json& config, Simulation* sim); // <-- NEW: Added Simulation pointer
    void solve(Mesh& mesh, int num_steps);

private:
    json solve_time_step(int step_num); // <-- NEW: Returns json metrics
    void check_failure_criterion(Node& node);
    double calculate_johnson_cook_stress(double temp, double strain);
    double calculate_temp_increase(double stress);
    double calculate_heat_dissipation(double current_temp);
    double calculate_usui_wear_rate(double temp, double stress); // <-- NEW: Wear model function

    json config_data;
    double dt;
    Mesh* active_mesh;
    Simulation* parent_sim; // <-- NEW: Pointer to parent sim for time-series output
};

class Simulation {
public:
    Simulation();
    void run();

    // NEW: Public member to allow FEASolver to write to it
    json time_series_output; 

private:
    void load_config(const std::string& path);
    void load_geometry();
    void write_output();

    Mesh mesh;
    json config_data;
};

#endif // SIMULATION_H
