#include <iostream>
#include "simulation.h"

int main() {
    try {
        // Create the simulation object. It will handle its own setup.
        Simulation sim;
        // Run the entire simulation process.
        sim.run();
    } catch (const std::exception& e) {
        // Catch any fatal errors that occurred during the simulation.
        std::cerr << "A fatal error occurred: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

