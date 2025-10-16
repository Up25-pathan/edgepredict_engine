#include "simulation.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <omp.h>
#include "stl_reader.h"
#include <algorithm>
#include <iomanip> // For std::setw

// --- FEASolver Class Implementation ---

FEASolver::FEASolver(const json& config) {
    config_data = config;
    dt = config_data["simulation_parameters"]["time_step_duration_s"].get<double>();
}

void FEASolver::solve(Mesh& mesh, int num_steps) {
    active_mesh = &mesh;
    std::cout << "  Running CPU-based FEA simulation for " << num_steps << " steps..." << std::endl;

    int thread_count = 0;
    #pragma omp parallel
    {
        #pragma omp single
        thread_count = omp_get_num_threads();
    }
    std::cout << "  - Activating OpenMP multithreading with " << thread_count << " threads." << std::endl;

    for (int i = 1; i <= num_steps; ++i) {
        solve_time_step(i);
    }
}

void FEASolver::solve_time_step(int step_num) {
    double max_temp = 0.0;
    int fractured_nodes_this_step = 0;
    double strain_increment = config_data["physics_parameters"]["strain_increment_per_step"].get<double>();

    #pragma omp parallel for reduction(max:max_temp) reduction(+:fractured_nodes_this_step)
    for (size_t i = 0; i < active_mesh->nodes.size(); ++i) {
        Node& node = active_mesh->nodes[i];
        
        if (node.status == NodeStatus::OK) {
            // 1. NEW: Accumulate plastic strain with each time step.
            node.strain += strain_increment;

            // 2. Calculate stress based on the node's current temperature AND accumulated strain.
            node.stress = calculate_johnson_cook_stress(node.temperature, node.strain);
            
            check_failure_criterion(node);

            if (node.status == NodeStatus::OK) {
                double heat_increase = calculate_temp_increase(node.stress);
                double heat_dissipation = calculate_heat_dissipation(node.temperature);
                node.temperature += (heat_increase - heat_dissipation) * dt;
            } else {
                fractured_nodes_this_step++;
            }
        }
        
        if (node.temperature > max_temp) {
            max_temp = node.temperature;
        }
    }

    std::cout << "  - Step " << step_num << "/" << config_data["simulation_parameters"]["num_steps"].get<int>()
              << " | Max Temp: " << std::fixed << std::setprecision(4) << max_temp << " C"
              << " | Fractured Nodes: " << fractured_nodes_this_step << std::endl;
}

void FEASolver::check_failure_criterion(Node& node) {
    const auto& failure_props = config_data["material_properties"]["failure_criterion"];
    double uts = failure_props["ultimate_tensile_strength_MPa"].get<double>();

    if (node.stress > uts) {
        node.status = NodeStatus::FRACTURED;
    }
}

// NEW: Stress calculation now also takes the node's specific strain as an input.
double FEASolver::calculate_johnson_cook_stress(double temp, double strain) {
    const auto& props = config_data["material_properties"];
    const auto& physics = config_data["physics_parameters"];
    double A = props["A_yield_strength_MPa"].get<double>();
    double B = props["B_strain_hardening_MPa"].get<double>();
    double n = props["n_strain_hardening_exp"].get<double>();
    double m = props["m_thermal_softening_exp"].get<double>();
    double T_room = physics["ambient_temperature_C"].get<double>();
    double T_melt = props["melting_point_C"].get<double>();

    if (temp > T_melt) temp = T_melt;
    if (temp < T_room) temp = T_room;

    double strain_hardening = B * pow(strain, n);
    double thermal_softening = 1.0 - pow((temp - T_room) / (T_melt - T_room), m);

    return (A + strain_hardening) * thermal_softening;
}

double FEASolver::calculate_temp_increase(double stress) {
    const auto& props = config_data["material_properties"];
    const auto& physics = config_data["physics_parameters"];
    double density = props["density_kg_m3"].get<double>();
    double specific_heat = props["specific_heat_J_kgC"].get<double>();
    double strain_rate = physics["strain_rate"].get<double>();
    return (stress * strain_rate) / (density * specific_heat);
}

double FEASolver::calculate_heat_dissipation(double current_temp) {
    const auto& params = config_data["physics_parameters"];
    double h = params["heat_transfer_coefficient"].get<double>();
    double T_ambient = params["ambient_temperature_C"].get<double>();
    
    double surface_area = 0.0001;
    double volume = 0.000001;
    double density = config_data["material_properties"]["density_kg_m3"].get<double>();
    double specific_heat = config_data["material_properties"]["specific_heat_J_kgC"].get<double>();

    double cooling_rate = (h * surface_area * (current_temp - T_ambient)) / (density * specific_heat * volume);
    return (cooling_rate > 0) ? cooling_rate : 0.0;
}


// --- Simulation Class Implementation ---

Simulation::Simulation() {
    try {
        load_config("input.json");
        load_geometry();
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR during initialization: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Simulation::run() {
    std::cout << "--- EdgePredict Core Simulation Engine v2.0 ---" << std::endl;
    int num_steps = config_data["simulation_parameters"]["num_steps"].get<int>();
    FEASolver solver(config_data);
    solver.solve(mesh, num_steps);
    write_output();
    std::cout << "--- Engine finished ---" << std::endl;
}

void Simulation::load_config(const std::string& path) {
    std::cout << "  - Attempting to load configuration from: \"" << path << "\"" << std::endl;
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open input file: " + path);
    }
    
    try {
        config_data = json::parse(f);
    } catch (json::parse_error& e) {
        throw std::runtime_error("Failed to parse input.json: " + std::string(e.what()));
    }
}

void Simulation::load_geometry() {
    std::string stl_path = config_data["file_paths"]["tool_geometry"].get<std::string>();
    std::cout << "  - Loading geometry from: \"" << stl_path << "\"" << std::endl;

    try {
        stl_reader::StlMesh<float, unsigned int> stl_mesh(stl_path);
        mesh.nodes.reserve(stl_mesh.num_vrts());
        double ambient_temp = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
        for (size_t i = 0; i < stl_mesh.num_vrts(); ++i) {
            const float* pos = stl_mesh.vrt_coords(i);
            Node node;
            node.position = {pos[0], pos[1], pos[2]};
            node.temperature = ambient_temp;
            mesh.nodes.push_back(node);
        }
        std::cout << "  - Loaded " << mesh.nodes.size() << " nodes from " << stl_path << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to read or parse STL file '" + stl_path + "': " + e.what());
    }
}

void Simulation::write_output() {
    std::string output_path = config_data["file_paths"]["output_results"].get<std::string>();
    std::cout << "Writing final results to: \"" << output_path << "\"" << std::endl;

    json output_data;
    output_data["num_nodes"] = mesh.nodes.size();
    for (const auto& node : mesh.nodes) {
        std::string status_str = (node.status == NodeStatus::OK) ? "OK" : "FRACTURED";
        
        json node_data;
        node_data["position"] = {node.position.x(), node.position.y(), node.position.z()};
        node_data["temperature_C"] = node.temperature;
        node_data["stress_MPa"] = node.stress;
        node_data["strain"] = node.strain; // Add accumulated strain to output
        node_data["status"] = status_str;
        output_data["nodes"].push_back(node_data);
    }

    std::ofstream o(output_path);
    o << std::setw(4) << output_data << std::endl;
}

