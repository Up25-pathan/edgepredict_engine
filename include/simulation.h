#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <string>
#include "json.hpp"
#include <Eigen/Dense>

// Use the nlohmann namespace for JSON
using json = nlohmann::json;

enum class NodeStatus {
    OK,
    FRACTURED
};

// Represents a single point (node) in the 3D mesh
struct Node {
    Eigen::Vector3d position;
    double temperature = 25.0;
    double stress = 0.0;
    double strain = 0.0; // NEW: Each node now tracks its own accumulated strain
    NodeStatus status = NodeStatus::OK;
};

// Represents the entire 3D mesh of the tool
struct Mesh {
    std::vector<Node> nodes;
};

// The core physics solver
class FEASolver {
public:
    FEASolver(const json& config);
    void solve(Mesh& mesh, int num_steps);

private:
    void solve_time_step(int step_num);

    // Physics calculation methods
    // NEW: Stress calculation now depends on the node's specific strain
    double calculate_johnson_cook_stress(double temp, double strain); 
    double calculate_temp_increase(double stress);
    double calculate_heat_dissipation(double current_temp);
    void check_failure_criterion(Node& node);

    Mesh* active_mesh = nullptr;
    json config_data;
    double dt; // Time step duration
};

// Main simulation orchestration class
class Simulation {
public:
    Simulation();
    void run();

private:
    void load_config(const std::string& path);
    void load_geometry();
    void write_output();

    json config_data;
    Mesh mesh;
};

#endif // SIMULATION_H

