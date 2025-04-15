# RL-Based Traffic Light Control System

An intelligent traffic light control system that uses reinforcement learning to optimize signal timings at intersections. The system adapts to real-time traffic conditions, reducing congestion and waiting times.

## Overview

This project implements a deep Q-learning (DQN) agent that controls traffic signals at a junction. The agent learns optimal signal timing strategies by observing traffic patterns and adjusting green light durations to minimize vehicle waiting times and improve overall throughput.

## Features

- **Adaptive Traffic Control**: Dynamically adjusts traffic light timings based on real-time traffic conditions
- **Deep Q-Learning**: Uses a deep neural network to learn optimal control policies
- **Flexible Configuration**: Supports various junction layouts and traffic patterns
- **Simulation Integration**: Fully integrated with SUMO (Simulation of Urban MObility)
- **Performance Metrics**: Tracks waiting times, throughput, and other key performance indicators

## Requirements

- Python 3.6+
- SUMO (Simulation of Urban MObility)
- PyTorch
- NumPy
- TraCI (Traffic Control Interface)

## Installation

1. **Install SUMO**
   
   Follow the installation instructions for your platform from the [SUMO website](https://sumo.dlr.de/docs/Installing.html).

2. **Set up the Python environment**

   ```bash
   pip install torch numpy
   ```

3. **Set the SUMO_HOME environment variable**

   For Windows:
   ```
   set SUMO_HOME=C:\path\to\sumo
   ```

   For Linux/Mac:
   ```
   export SUMO_HOME=/path/to/sumo
   ```

4. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/traffic-rl-control.git
   cd traffic-rl-control
   ```

## Usage

1. **Run the simulation with default parameters**

   ```bash
   python model.py
   ```

2. **Run with GUI**

   ```bash
   python model.py -g
   ```

3. **Customize simulation parameters**

   ```bash
   python model.py -c net.sumocfg -e 20 -s 5000
   ```

   Options:
   - `-c`, `--config`: SUMO configuration file (default: "net.sumocfg")
   - `-g`, `--gui`: Run SUMO with GUI
   - `-e`, `--episodes`: Number of training episodes (default: 10)
   - `-s`, `--steps`: Steps per episode (default: 3600)

## How It Works

The system uses a Deep Q-Network (DQN) to learn the optimal traffic light control policy:

1. **State Representation**: The state includes vehicle counts and waiting times for each approach.
2. **Actions**: The agent can maintain, extend, or reduce green light durations.
3. **Reward Function**: The agent receives rewards based on reductions in total waiting time.
4. **Learning**: The agent learns from experience using replay memory and target networks.

## Project Structure

- `model.py`: Main DQN implementation and simulation runner
- `net.rou.xml`: Vehicle route definitions
- `net.sumocfg`: SUMO configuration file
- `best_model.pth`: Saved model weights (generated after training)

## Future Improvements

- Multi-junction coordination
- Support for more complex traffic patterns
- Integration with real-world traffic data
- Comparison with other RL algorithms

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SUMO team for the traffic simulation environment
- PyTorch community for the deep learning framework 