# Use a single, consistent base image for both building and running.
# The official GCC image has all the tools and libraries we need, perfectly matched.
FROM gcc:latest

# Install the Eigen library, a required dependency for our C++ code.
RUN apt-get update && apt-get install -y libeigen3-dev && rm -rf /var/lib/apt/lists/*

# Set the working directory for the build.
WORKDIR /usr/src/app

# Copy all local project files into the container.
COPY . .

# Compile the C++ source code and place the final executable in a system-wide path.
# This ensures the program is always available, separate from any data directories.
RUN g++ -I./include -I /usr/include/eigen3 -std=c++17 -O3 -fopenmp src/main.cpp src/simulation.cpp -o /usr/local/bin/engine

# Set the final working directory for the running container to a dedicated data folder.
WORKDIR /data

# The default command to run when the container starts. It will execute the 'engine'
# command from the system path, and it will run inside this '/data' directory.
CMD ["engine"]