#include "simulation.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <omp.h>
#include "stl_reader.h"
#include <algorithm>
#include <iomanip> // For std::setw

// --- FEASolver Class Implementation ---

// NEW: Constructor now accepts parent Simulation pointer
FEASolver::FEASolver(const json& config, Simulation* sim) {
    config_data = config;
    parent_sim = sim; // Store the pointer
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
        // NEW: solve_time_step now returns metrics, which we store
        json step_metrics = solve_time_step(i);
        step_metrics["step"] = i;
        parent_sim->time_series_output.push_back(step_metrics);
    }
}

// NEW: This function now returns JSON and calculates wear + more metrics
json FEASolver::solve_time_step(int step_num) {
    double max_temp = 0.0;
    double max_stress = 0.0; // NEW: Track max stress
    double total_wear = 0.0;   // NEW: Track total wear
    int fractured_nodes_this_step = 0;
    double strain_increment = config_data["physics_parameters"]["strain_increment_per_step"].get<double>();

    // NEW: Added max_stress and total_wear to the reduction
    #pragma omp parallel for reduction(max:max_temp, max_stress) reduction(+:fractured_nodes_this_step, total_wear)
    for (size_t i = 0; i < active_mesh->nodes.size(); ++i) {
        Node& node = active_mesh->nodes[i];
        
        if (node.status == NodeStatus::OK) {
            // 1. Accumulate plastic strain
            node.strain += strain_increment;

            // 2. Calculate stress
            node.stress = calculate_johnson_cook_stress(node.temperature, node.strain);
            
            // 3. (NEW) Calculate and accumulate wear
            // This happens *before* failure check, as wear occurs up to the point of fracture
            double wear_rate = calculate_usui_wear_rate(node.temperature, node.stress);
            node.accumulated_wear += wear_rate * dt;

            // 4. Check failure criterion
            check_failure_criterion(node);

            // 5. Calculate temperature (if not fractured)
            if (node.status == NodeStatus::OK) {
                double heat_increase = calculate_temp_increase(node.stress);
                double heat_dissipation = calculate_heat_dissipation(node.temperature);
                node.temperature += (heat_increase - heat_dissipation) * dt;
            } else {
                fractured_nodes_this_step++;
            }
        }
        
        // --- Metric tracking ---
        if (node.temperature > max_temp) {
            max_temp = node.temperature;
        }
        // NEW: Track max stress
        if (node.stress > max_stress) {
            max_stress = node.stress;
        }
        // NEW: Track total accumulated wear
        total_wear += node.accumulated_wear;
    }

    std::cout << "  - Step " << step_num << "/" << config_data["simulation_parameters"]["num_steps"].get<int>()
              << " | Max Temp: " << std::fixed << std::setprecision(4) << max_temp << " C"
              << " | Max Stress: " << std::fixed << std::setprecision(2) << max_stress << " MPa" // NEW
              << " | Fractured Nodes: " << fractured_nodes_this_step << std::endl;

    // NEW: Return step metrics as a JSON object
    json metrics;
    metrics["max_temperature_C"] = max_temp;
    metrics["max_stress_MPa"] = max_stress;
    metrics["total_accumulated_wear"] = total_wear;
    metrics["fractured_nodes_count"] = fractured_nodes_this_step;
    return metrics;
}

// --- UPDATED: Real Usui Wear Model Implementation ---
double FEASolver::calculate_usui_wear_rate(double temp, double stress) {
    // Per the roadmap, we must implement the Usui wear model.
    // This requires new parameters from input.json.
    try {
        // Get parameters from the "usui_wear_model" object in the JSON
        const auto& wear_props = config_data["material_properties"]["usui_wear_model"];
        double A = wear_props["A_constant"].get<double>();
        double B_inv_temp = wear_props["B_inv_temp_K"].get<double>(); // (1/T) coefficient

        // Get the sliding velocity from the physics parameters
        double V_slide = config_data["physics_parameters"]["sliding_velocity_m_s"].get<double>();

        double temp_K = temp + 273.15; // Convert C to Kelvin

        // Basic check for valid physics
        if (temp_K <= 0 || stress <= 0) return 0.0;

        // Basic Usui Wear Model: 
        // Wear_Rate (m/s) = A * Stress(MPa) * Sliding_Velocity(m/s) * exp(-B / Temp_K)
        // Note: The 'A' constant must have the correct units (e.g., 1/MPa) for this to be m/s
        return A * stress * V_slide * exp(-B_inv_temp / temp_K);

    } catch (const std::exception& e) {
        // If "usui_wear_model" or its keys aren't in input.json,
        // print a warning and return zero wear.
        // This prevents a crash if the user forgets to add them.
        
        // Use #pragma omp critical to prevent garbled output from multiple threads
        #pragma omp critical
        {
            static bool warning_shown = false;
            if (!warning_shown) {
                std::cerr << "Warning: Usui wear parameters not found in input.json. Returning 0 wear." << std::endl;
                warning_shown = true;
            }
        }
        return 0.0;
    }
}
// --- End of update ---


void FEASolver::check_failure_criterion(Node& node) {
    const auto& failure_props = config_data["material_properties"]["failure_criterion"];
    double uts = failure_props["ultimate_tensile_strength_MPa"].get<double>();

    if (node.stress > uts) {
        node.status = NodeStatus::FRACTURED;
    }
}

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
        time_series_output = json::array(); // NEW: Initialize the time-series array
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
    
    // NEW: Pass 'this' (the simulation instance) to the solver
    FEASolver solver(config_data, this); 
    
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

    // NEW: Add the completed time-series data to the output
    output_data["time_series_data"] = time_series_output;

    // Save final node state (as before)
    output_data["nodes"] = json::array(); // Initialize 'nodes' as an array
    for (const auto& node : mesh.nodes) {
        std::string status_str = (node.status == NodeStatus::OK) ? "OK" : "FRACTURED";
        
        json node_data;
        node_data["position"] = {node.position.x(), node.position.y(), node.position.z()};
        node_data["temperature_C"] = node.temperature;
        node_data["stress_MPa"] = node.stress;
        node_data["strain"] = node.strain;
        node_data["accumulated_wear"] = node.accumulated_wear; // NEW: Add final wear to output
        node_data["status"] = status_str;
        output_data["nodes"].push_back(node_data);
    }

    std::ofstream o(output_path);
    o << std::setw(4) << output_data << std::endl;
}
