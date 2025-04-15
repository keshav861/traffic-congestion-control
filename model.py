from __future__ import absolute_import, print_function
import os
import sys
import time
import optparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Error: Please set the SUMO_HOME environment variable.")

from sumolib import checkBinary
import traci

# Define the DQN architecture
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
        
    def push(self, experience):
        self.memory.append(experience)
        
    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return random.sample(self.memory, len(self.memory))
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Define the RL Agent
class RLTrafficAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        
        # Initialize networks
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory()
        self.batch_size = 64
        
    def remember(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
      #selects actions using ε-greedy policy  
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_values = self.policy_net(state_tensor)
            return torch.argmax(action_values).item()
    # implements the DQN learning algorithm
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([exp.state for exp in experiences])
        actions = torch.LongTensor([[exp.action] for exp in experiences])
        rewards = torch.FloatTensor([[exp.reward] for exp in experiences])
        next_states = torch.FloatTensor([exp.next_state for exp in experiences])
        dones = torch.FloatTensor([[exp.done] for exp in experiences])
        
        # Get current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        # Calculate target Q values
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Calculate loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)
        
    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Traffic Light Controller with reinforcement learning
class IntelligentTrafficController:
    def __init__(self, junction_id, base_green_time=60, update_window=20):
        self.junction_id = junction_id
        
        # Lane groups by approach direction
        self.directions = ["east", "west", "south", "north"]
        self.incoming_lanes = {
            "west": ["B0A0_0"],     # From B0
            "north": ["A1A0_0"],    # From A1
            "east": ["-E1_0"],      # From J1
            "south": ["-E0_0"]      # From J0
        }
        
        # Verify lane IDs exist in network
        self.verify_lanes()
        
        # Signal timing parameters
        self.base_green_time = base_green_time
        self.update_window = update_window
        self.time_adjustment = 10
        self.yellow_time = 3  # Yellow phase duration
        
        # State tracking
        self.current_direction_index = 0
        self.direction_sequence = self.directions.copy()  # Initialize with default sequence
        self.time_in_phase = 0
        self.phase_duration = self.base_green_time
        self.phase_extended = False
        self.phase_reduced = False
        self.cycle_completed = False
        
        # Define phase states for each direction
        self.phase_states = {
            "east": "rrrrrrGGGrrr",   # East approach green
            "west": "GGGrrrrrrrrr",   # West approach green
            "south": "rrrGGGrrrrrr",  # South approach green
            "north": "rrrrrrrrrGGG"   # North approach green 
        }
        
        # Yellow phase states for transitions
        self.yellow_states = {
            "east": "rrrrrryyyrrr",
            "west": "yyyrrrrrrrrr",
            "south": "rrryyyrrrrrr",
            "north": "rrrrrrrrryyy"
        }
        
        # RL agent parameters
        self.state_size = 8  # 4 directions x 2 features (vehicle count, waiting time)
        self.action_size = 3  # 0: no change, 1: extend, 2: reduce
        self.agent = RLTrafficAgent(self.state_size, self.action_size)
        
        # For reward calculation
        self.previous_waiting_time = 0
        self.current_state = None
        self.last_action = None
        
        # For logging
        self.total_rewards = 0
        self.episode_count = 0
        
    def verify_lanes(self):
        """Verify that configured lanes exist in the SUMO network"""
        try:
            all_lanes = traci.lane.getIDList()
            print(f"Available lanes in network: {all_lanes}")
            
            missing_lanes = []
            for direction, lanes in self.incoming_lanes.items():
                for lane in lanes:
                    if lane not in all_lanes:
                        missing_lanes.append(f"{lane} (direction: {direction})")
            
            if missing_lanes:
                print(f"WARNING: The following lanes are not found in the network: {missing_lanes}")
                print("This may cause errors during simulation")
                
        except Exception as e:
            print(f"Error verifying lanes: {e}")
        
    def get_lane_vehicle_count(self, lane_id):
        """Get the number of vehicles on a lane"""
        try:
            return len(traci.lane.getLastStepVehicleIDs(lane_id))
        except traci.exceptions.TraCIException as e:
            print(f"Error getting vehicle count for lane {lane_id}: {e}")
            return 0
    
    def get_lane_waiting_time(self, lane_id):
        """Get the total waiting time of vehicles on a lane"""
        try:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            total_waiting_time = 0
            for vehicle in vehicles:
                total_waiting_time += traci.vehicle.getWaitingTime(vehicle)
            return total_waiting_time
        except traci.exceptions.TraCIException as e:
            print(f"Error getting waiting time for lane {lane_id}: {e}")
            return 0
    
    def get_direction_metrics(self, direction):
        """Get vehicle count and waiting time for a direction"""
        try:
            lanes = self.incoming_lanes[direction]
            total_count = sum(self.get_lane_vehicle_count(lane) for lane in lanes)
            total_waiting_time = sum(self.get_lane_waiting_time(lane) for lane in lanes)
            return total_count, total_waiting_time
        except Exception as e:
            print(f"Error getting metrics for direction {direction}: {e}")
            return 0, 0
    
    def get_state(self):
        """Get the current state for the RL agent"""
        state = []
        for direction in self.directions:
            count, waiting_time = self.get_direction_metrics(direction)
            # Normalize the values
            normalized_count = min(count / 20.0, 1.0)  # Assuming max 20 vehicles
            normalized_waiting = min(waiting_time / 300.0, 1.0)  # Assuming max 300 seconds
            state.extend([normalized_count, normalized_waiting])
        return np.array(state)
    
    def calculate_reward(self):
        """Calculate reward based on the reduction in total waiting time"""
        current_waiting_time = 0
        for direction in self.directions:
            _, waiting_time = self.get_direction_metrics(direction)
            current_waiting_time += waiting_time
        
        # Reward is negative of the change in waiting time
        reward = self.previous_waiting_time - current_waiting_time
        
        # Additional reward/penalty for specific actions
        if self.last_action == 1 and self.phase_extended:  # Extended green
            # If many vehicles passed during extension, give bonus
            if reward > 10:
                reward += 5
        elif self.last_action == 2 and self.phase_reduced:  # Reduced green
            # If waiting time didn't increase much after reduction, give bonus
            if reward > -5:
                reward += 3
        
        self.previous_waiting_time = current_waiting_time
        return reward
    
    def determine_direction_sequence(self):
        """Determine direction sequence based on traffic density"""
        try:
            # Get metrics for each direction
            metrics = {}
            for direction in self.directions:
                count, waiting_time = self.get_direction_metrics(direction)
                # Score based on vehicle count and waiting time
                score = count + (waiting_time / 10.0)
                metrics[direction] = score
            
            # Sort directions by score (descending)
            self.direction_sequence = sorted(self.directions, key=lambda x: metrics[x], reverse=True)
            print(f"Direction sequence: {self.direction_sequence}")
            
            # Safety check - if sequence is empty, use default order
            if not self.direction_sequence:
                print("Warning: Direction sequence is empty, using default order")
                self.direction_sequence = self.directions.copy()
                
            # Start with the highest priority direction
            self.current_direction_index = 0
            self.cycle_completed = False
        except Exception as e:
            print(f"Error in determine_direction_sequence: {e}")
            # Use default direction sequence as fallback
            self.direction_sequence = self.directions.copy()
            self.current_direction_index = 0
            self.cycle_completed = False
    
    def set_traffic_light_phase(self, direction, is_yellow=False):
        """Set the traffic light phase for a direction"""
        try:
            if is_yellow:
                traci.trafficlight.setRedYellowGreenState(self.junction_id, self.yellow_states[direction])
            else:
                traci.trafficlight.setRedYellowGreenState(self.junction_id, self.phase_states[direction])
        except traci.exceptions.TraCIException as e:
            print(f"Error setting traffic light phase for {direction}: {e}")
    
    def start_new_green_phase(self):
        """Start a new green phase for the current direction"""
        try:
            # Safety check
            if not self.direction_sequence or self.current_direction_index >= len(self.direction_sequence):
                print("Warning: Invalid direction sequence or index, resetting")
                self.direction_sequence = self.directions.copy()
                self.current_direction_index = 0
                
            current_direction = self.direction_sequence[self.current_direction_index]
            self.set_traffic_light_phase(current_direction)
            self.phase_duration = self.base_green_time
            self.time_in_phase = 0
            self.phase_extended = False
            self.phase_reduced = False
            print(f"Starting green phase for {current_direction}, duration: {self.phase_duration}s")
        except Exception as e:
            print(f"Error starting new green phase: {e}")
            # Default to first direction as fallback
            self.direction_sequence = self.directions.copy()
            self.current_direction_index = 0
            if self.directions:
                self.set_traffic_light_phase(self.directions[0])
    
    def move_to_next_direction(self):
        """Move to the next direction in the sequence with a yellow transition"""
        try:
            # Safety check
            if not self.direction_sequence or self.current_direction_index >= len(self.direction_sequence):
                print("Warning: Invalid direction sequence or index while moving to next direction, resetting")
                self.direction_sequence = self.directions.copy()
                self.current_direction_index = 0
                
            current_direction = self.direction_sequence[self.current_direction_index]
            
            # Set yellow phase
            self.set_traffic_light_phase(current_direction, is_yellow=True)
            
            # Update direction index
            self.current_direction_index = (self.current_direction_index + 1) % len(self.direction_sequence)
            
            # Check if we've completed a cycle
            if self.current_direction_index == 0:
                self.cycle_completed = True
                
            # We'll start the next green phase after yellow
            self.time_in_phase = 0
        except Exception as e:
            print(f"Error moving to next direction: {e}")
            # Reset as fallback
            self.direction_sequence = self.directions.copy()
            self.current_direction_index = 0
            self.time_in_phase = 0
    
    def adjust_green_time(self, action):
        """Adjust green time based on the RL agent's action"""
        try:
            if action == 0:  # No change
                return
            
            if action == 1 and not self.phase_extended:  # Extend
                self.phase_duration += self.time_adjustment
                self.phase_extended = True
                print(f"  Extended green time to {self.phase_duration}s")
                return
            
            if action == 2 and not self.phase_reduced:  # Reduce
                self.phase_duration = max(self.base_green_time - self.time_adjustment, 
                                      self.base_green_time - self.update_window)
                self.phase_reduced = True
                print(f"  Reduced green time to {self.phase_duration}s")
                return
        except Exception as e:
            print(f"Error adjusting green time: {e}")
    
    def update(self, step):
        """Update traffic light controller"""
        try:
            print(f"Step {step}: Updating traffic light controller")
            
            # Debug output to trace the error
            if step == 0:
                print(f"Initial direction_sequence: {self.direction_sequence}")
                print(f"Current index: {self.current_direction_index}")
                print(f"Time in phase: {self.time_in_phase}")
                
            # If we've completed a cycle, re-evaluate direction sequence
            if self.cycle_completed and self.time_in_phase == 0:
                print("Re-evaluating direction sequence")
                self.determine_direction_sequence()
            
            # Get current state
            current_state = self.get_state()
            print(f"Current state: {current_state}")
            
            # If we're just starting
            if self.time_in_phase == 0:
                if step > 0:  # Not the first step of simulation
                    # Calculate reward for previous action
                    if self.current_state is not None:
                        reward = self.calculate_reward()
                        self.total_rewards += reward
                        print(f"Reward: {reward}, Total rewards: {self.total_rewards}")
                        
                        # Store experience in replay memory
                        done = self.cycle_completed
                        self.agent.remember(self.current_state, self.last_action, 
                                          reward, current_state, done)
                        
                        # Learn from experiences
                        self.agent.replay()
                        
                        if done:
                            self.episode_count += 1
                            print(f"Episode {self.episode_count}, Total Reward: {self.total_rewards}")
                            self.total_rewards = 0
                            
                            # Update target network every 5 episodes
                            if self.episode_count % 5 == 0:
                                self.agent.update_target_network()
                
                # Start new green phase
                self.start_new_green_phase()
                
            # At decision time (when we're near the update window threshold)
            elif self.time_in_phase == self.base_green_time - self.update_window:
                # Get state and choose action
                self.current_state = current_state
                action = self.agent.act(current_state)
                self.last_action = action
                print(f"Chose action: {action} (0=no change, 1=extend, 2=reduce)")
                
                # Adjust green time based on action
                self.adjust_green_time(action)
            
            # Increment time in phase
            self.time_in_phase += 1
            
            # Check if it's time to end the phase
            if self.time_in_phase >= self.phase_duration:
                print(f"Phase duration reached ({self.phase_duration}s), moving to next direction")
                self.move_to_next_direction()
                
                # Set temporary phase duration for yellow phase
                self.phase_duration = self.yellow_time
                
        except Exception as e:
            print(f"Error in update method: {e}")
            import traceback
            traceback.print_exc()
            # Try to recover
            self.direction_sequence = self.directions.copy()
            self.current_direction_index = 0
            self.time_in_phase = 0
            self.phase_duration = self.base_green_time

# SUMO Runner with RL integration
class SUMOSimulation:
    def __init__(self, sumo_cfg, gui=False, episodes=10, steps_per_episode=3600):
        self.sumo_cfg = sumo_cfg
        self.gui = gui
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        
        # Define junction ID based on the actual network
        self.junction_id = "A0"  # Updated to match your network file
        
        # For stats tracking
        self.total_waiting_times = []
        self.total_throughputs = []
        
        # No vehicle detection settings
        self.no_vehicles_threshold = 10  # Number of consecutive steps with zero vehicles to stop simulation
        self.no_vehicles_counter = 0

    def run(self):
        # Start SUMO
        if self.gui:
            sumo_binary = checkBinary('sumo-gui')
        else:
            sumo_binary = checkBinary('sumo')
        
        for episode in range(self.episodes):
            print(f"\nStarting Episode {episode+1}/{self.episodes}")
            
            try:
                # Start the simulation
                traci.start([sumo_binary, "-c", self.sumo_cfg])
                
                # Print traffic light info
                print("Traffic light info:")
                print(f"Junction ID: {self.junction_id}")
                if self.junction_id in traci.trafficlight.getIDList():
                    print(f"Current program: {traci.trafficlight.getProgram(self.junction_id)}")
                    print(f"Phase: {traci.trafficlight.getPhase(self.junction_id)}")
                    print(f"State: {traci.trafficlight.getRedYellowGreenState(self.junction_id)}")
                else:
                    print(f"Warning: Junction {self.junction_id} not found in traffic light IDs: {traci.trafficlight.getIDList()}")
                
                # Create traffic controller AFTER connecting to SUMO
                self.controller = IntelligentTrafficController(self.junction_id)
                
                # Stats for this episode
                episode_waiting_time = 0
                episode_throughput = 0
                
                # Reset no vehicles counter
                self.no_vehicles_counter = 0
                
                # Simulation loop
                step = 0
                while step < self.steps_per_episode:
                    try:
                        traci.simulationStep()
                        self.controller.update(step)
                        step += 1
                        
                        # Check if there are any vehicles on the road
                        total_vehicles = traci.vehicle.getIDCount()
                        
                        if total_vehicles == 0:
                            self.no_vehicles_counter += 1
                            if self.no_vehicles_counter >= self.no_vehicles_threshold:
                                print(f"No vehicles detected for {self.no_vehicles_threshold} consecutive steps. Ending simulation.")
                                break
                        else:
                            # Reset counter if vehicles are detected
                            self.no_vehicles_counter = 0
                        
                        # Collect stats every 100 steps
                        if step % 100 == 0:
                            # Calculate total waiting time
                            total_waiting = 0
                            for direction in self.controller.directions:
                                for lane in self.controller.incoming_lanes[direction]:
                                    for veh in traci.lane.getLastStepVehicleIDs(lane):
                                        total_waiting += traci.vehicle.getWaitingTime(veh)
                            
                            episode_waiting_time += total_waiting
                            
                            # Calculate throughput (vehicles that have reached their destination)
                            arrived = traci.simulation.getArrivedNumber()
                            episode_throughput += arrived
                            
                            print(f"Step {step}/{self.steps_per_episode}: "
                                f"Waiting time: {total_waiting}, Arrived vehicles: {arrived}, "
                                f"Vehicles on road: {total_vehicles}")
                    except Exception as e:
                        print(f"Error in simulation step {step}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Try to continue with next step
                        continue
                
                # Record episode stats if we completed at least some steps
                if step > 0:
                    avg_waiting_time = episode_waiting_time / max(1, (step // 100))
                    self.total_waiting_times.append(avg_waiting_time)
                    self.total_throughputs.append(episode_throughput)
                    
                    # Save the model if it's the best so far (based on throughput)
                    if episode > 0 and episode_throughput > max(self.total_throughputs[:-1]):
                        self.controller.agent.save("best_model.pth")
                        print("Saved best model so far")
                    
                    print(f"Episode {episode+1} completed. "
                        f"Avg waiting time: {avg_waiting_time:.2f}, "
                        f"Total throughput: {episode_throughput}")
                    
                    # If simulation ended early due to no vehicles, notify user
                    if self.no_vehicles_counter >= self.no_vehicles_threshold:
                        print(f"Episode ended early at step {step} due to no vehicles on the road.")
            
            except Exception as e:
                print(f"Error during simulation: {e}")
                import traceback
                traceback.print_exc()
            finally:
                try:
                    traci.close()
                except:
                    pass
        
        print("\nTraining completed!")
        
        # Calculate final stats safely
        if self.total_waiting_times:
            print(f"Final average waiting time: {np.mean(self.total_waiting_times):.2f}")
        else:
            print("No valid waiting time data collected")
            
        if self.total_throughputs:
            print(f"Final average throughput: {np.mean(self.total_throughputs):.2f}")
        else:
            print("No valid throughput data collected")
            
        print("Model saved as 'best_model.pth'")

# Option parser
def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("-c", "--config", dest="sumocfg", help="SUMO config file", default="net.sumocfg")
    optParser.add_option("-g", "--gui", action="store_true", dest="gui", default=False, help="Run SUMO with GUI")
    optParser.add_option("-e", "--episodes", dest="episodes", type="int", default=10, help="Number of episodes")
    optParser.add_option("-s", "--steps", dest="steps", type="int", default=3600, help="Steps per episode")
    return optParser.parse_args()[0]

if __name__ == '__main__':
    options = get_options()
    
    # Create and run simulation
    sim = SUMOSimulation(options.sumocfg, gui=options.gui, 
                         episodes=options.episodes, 
                         steps_per_episode=options.steps)
    sim.run()