"""
Multi-Agent Racing Safe MetaDrive Environment

This environment extends the SafeMetaDriveEnv to support Multi-Agent Reinforcement Learning (MARL)
between two cars in a racing scenario. It incorporates:
1. Concurrent control of two vehicles
2. Race-based reward structure (leading car gets bonus)
3. Support for reverse driving
4. Continuous simulation despite collisions or boundary violations
"""

import math
import numpy as np
from typing import Dict, Any, Tuple

from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network import Road
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.utils import Config, clip

# Define constants for the racing environment
RACING_SAFE_METADRIVE_DEFAULT_CONFIG = dict(
    # ===== Multi-agent settings =====
    is_multi_agent=True,
    num_agents=2,  # Fixed at 2 for racing scenario
    allow_respawn=False,  # No respawning in racing

    # Ensure we're using the correct agent IDs

    # ===== Racing settings =====
    # Reward for being in the lead
    leading_reward_factor=0.1,  # Reward multiplier for being in the lead
    winning_reward=10.0,  # Additional reward for winning the race
    checkpoint_reward=2.0,  # Reward for reaching a checkpoint
    first_checkpoint_bonus=1.0,  # Additional bonus for being first to reach a checkpoint
    num_checkpoints=5,  # Number of checkpoints to place around the track

    # ===== Safe driving settings =====
    # Override termination conditions to allow continuous simulation
    crash_vehicle_done=False,
    crash_object_done=False,
    out_of_road_done=False,

    # ===== Cost settings =====
    crash_vehicle_cost=10.0,
    crash_object_cost=5.0,
    out_of_road_cost=20.0,  # Increased to provide stronger incentive for staying on road
    on_yellow_line_cost=3.0,  # Cost for driving on yellow line (wrong side of road)
    on_white_line_cost=1.0,  # Cost for driving on white line

    # ===== Vehicle settings =====
    vehicle_config=dict(
        enable_reverse=True,  # Enable reverse driving
        vehicle_model="static_default",  # Use static model for consistent behavior
    ),

    # Agent configuration is handled in _post_process_config
)


class MultiAgentRacingSafeEnv(MultiAgentMetaDrive):
    """
    A Multi-Agent Racing environment based on SafeMetaDriveEnv.

    This environment features:
    1. Two cars racing against each other
    2. Modified reward structure to incentivize racing
    3. Continuous simulation despite collisions or boundary violations
    4. Support for reverse driving
    """

    def default_config(self) -> Config:
        # Start with SafeMetaDriveEnv config
        config = super(MultiAgentRacingSafeEnv, self).default_config()
        # Update with racing-specific settings
        config.update(RACING_SAFE_METADRIVE_DEFAULT_CONFIG, allow_add_new_key=True)
        return config

    def _post_process_config(self, config):
        # Process config using parent method first
        config = super(MultiAgentRacingSafeEnv, self)._post_process_config(config)

        # Ensure we have agent_configs for agent0 and agent1
        if "agent_configs" not in config:
            config["agent_configs"] = {}

        # Create agent configs if not present
        for i in range(2):  # Fixed at 2 agents for racing
            agent_id = f"agent{i}"
            if agent_id not in config["agent_configs"]:
                config["agent_configs"][agent_id] = {}

        # Set spawn lanes if not specified
        if "spawn_lane_index" not in config["agent_configs"]["agent0"]:
            config["agent_configs"]["agent0"]["spawn_lane_index"] = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)

        if "spawn_lane_index" not in config["agent_configs"]["agent1"]:
            config["agent_configs"]["agent1"]["spawn_lane_index"] = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 1)

        # Set special colors for better visibility
        config["agent_configs"]["agent0"]["use_special_color"] = True
        config["agent_configs"]["agent1"]["use_special_color"] = True

        return config

    def __init__(self, config=None):
        # Initialize with MultiAgentMetaDrive
        super(MultiAgentRacingSafeEnv, self).__init__(config)

        # Initialize episode cost tracking for each agent
        self.episode_cost = {}

        # Track agent progress for racing rewards
        self.agent_progress = {}
        self.agent_positions = {}
        self.race_finished = False
        self.race_winner = None

        # Initialize checkpoint tracking
        self.checkpoints = []
        self.agent_checkpoints = {}
        self.checkpoint_first_agent = {}

        # Initialize agent-specific attributes
        self._init_agent_attributes()

    def _init_agent_attributes(self):
        """Initialize agent-specific attributes."""
        # Get agent IDs from config
        agent_ids = list(self.config["agent_configs"].keys())
        if not agent_ids:
            agent_ids = ["agent0", "agent1"]  # Default if not specified

        # Initialize tracking dictionaries for each agent
        for agent_id in agent_ids:
            self.episode_cost[agent_id] = 0.0
            self.agent_progress[agent_id] = 0.0
            self.agent_positions[agent_id] = (0, 0)

    def reset(self, *args, **kwargs):
        """Reset the environment and initialize agent tracking."""
        # Reset episode costs and race status
        self._init_agent_attributes()
        self.race_finished = False
        self.race_winner = None

        # Initialize off-road and wrong-side tracking
        self.off_road_counter = {}
        self.off_road_status = {}
        self.wrong_side_counter = {}
        self.wrong_side_status = {}

        # Initialize checkpoint tracking
        self.agent_checkpoints = {}
        self.checkpoint_first_agent = {}

        # Reset the environment
        obs, info = super(MultiAgentRacingSafeEnv, self).reset(*args, **kwargs)

        # Initialize agent positions and tracking variables after reset
        for agent_id, agent in self.agents.items():
            self.agent_positions[agent_id] = agent.position
            self.off_road_counter[agent_id] = 0
            self.off_road_status[agent_id] = False
            self.wrong_side_counter[agent_id] = 0
            self.wrong_side_status[agent_id] = False
            self.agent_checkpoints[agent_id] = set()  # Track which checkpoints this agent has passed

        # Create checkpoints along the track
        self._create_checkpoints()

        return obs, info

    def step(self, actions):
        """
        Step the environment with actions from all agents.

        Args:
            actions: Dict mapping agent_ids to their respective actions

        Returns:
            obs: Dict of observations for each agent
            rewards: Dict of rewards for each agent
            terminated: Dict indicating if each agent is terminated
            truncated: Dict indicating if each agent is truncated
            info: Dict of additional info for each agent
        """
        # Initialize checkpoint info for this step
        self.checkpoint_info = {agent_id: [] for agent_id in self.agents.keys()}

        # Step the environment
        obs, rewards, terminated, truncated, info = super(MultiAgentRacingSafeEnv, self).step(actions)

        # Update agent progress for racing rewards
        self._update_agent_progress()

        # Apply racing rewards
        rewards = self._apply_racing_rewards(rewards, info)

        # Apply checkpoint rewards
        rewards = self._apply_checkpoint_rewards(rewards, info)

        # Check for race completion
        self._check_race_completion(terminated, info)

        # Add race information to info dict
        for agent_id in self.agents.keys():
            if agent_id in info:
                info[agent_id].update({
                    "progress": self.agent_progress[agent_id],
                    "is_leading": self._is_agent_leading(agent_id),
                    "race_finished": self.race_finished,
                    "race_winner": self.race_winner,
                    "checkpoints_passed": len(self.agent_checkpoints.get(agent_id, set())),
                    "total_checkpoints": len(self.checkpoints)
                })

                # Add checkpoint info if available
                if agent_id in self.checkpoint_info and self.checkpoint_info[agent_id]:
                    info[agent_id]["checkpoint_reached"] = True
                    info[agent_id]["checkpoint_info"] = self.checkpoint_info[agent_id]

        return obs, rewards, terminated, truncated, info

    def _create_checkpoints(self):
        """Create checkpoints along the track."""
        self.checkpoints = []
        self.checkpoint_first_agent = {}

        # Get the map and road network
        if not hasattr(self.engine, 'current_map') or not self.engine.current_map:
            return

        # Get the road network from the map
        road_network = self.engine.current_map.road_network
        if not road_network or not hasattr(road_network, 'graph'):
            return

        # Get all lanes from the road network
        all_lanes = []
        for from_node, to_nodes in road_network.graph.items():
            for to_node, lanes in to_nodes.items():
                all_lanes.extend(lanes)

        if not all_lanes:
            return

        # Create evenly spaced checkpoints along the track
        num_checkpoints = self.config["num_checkpoints"]
        checkpoint_lanes = []

        # Try to select lanes that are evenly distributed around the track
        if len(all_lanes) >= num_checkpoints:
            step = len(all_lanes) // num_checkpoints
            for i in range(0, len(all_lanes), step):
                if len(checkpoint_lanes) < num_checkpoints:
                    checkpoint_lanes.append(all_lanes[i])
        else:
            # If not enough lanes, use all available lanes
            checkpoint_lanes = all_lanes

        # Create checkpoints at the middle of each selected lane
        for i, lane in enumerate(checkpoint_lanes):
            checkpoint_id = i
            checkpoint_pos = lane.position(lane.length / 2, 0)
            self.checkpoints.append({
                "id": checkpoint_id,
                "position": checkpoint_pos,
                "lane": lane,
                "radius": 5.0  # Detection radius
            })
            self.checkpoint_first_agent[checkpoint_id] = None

        print(f"Created {len(self.checkpoints)} checkpoints along the track")

    def _update_agent_progress(self):
        """Update the progress of each agent along the track and check for checkpoints."""
        for agent_id, agent in self.agents.items():
            current_position = agent.position
            if agent_id in self.agent_positions:
                prev_position = self.agent_positions[agent_id]
                # Calculate distance traveled along the road
                if hasattr(agent, 'navigation') and agent.navigation and agent.navigation.current_ref_lanes:
                    # Use road direction to measure progress
                    lane = agent.navigation.current_ref_lanes[-1]
                    road_direction = lane.heading_theta_at(lane.local_coordinates(current_position)[0])
                    direction_vector = (math.cos(road_direction), math.sin(road_direction))
                    # Project movement onto road direction
                    movement = (current_position[0] - prev_position[0], current_position[1] - prev_position[1])
                    progress = movement[0] * direction_vector[0] + movement[1] * direction_vector[1]
                    self.agent_progress[agent_id] += progress

                    # Check if agent has reached any checkpoints
                    self._check_checkpoints(agent_id, current_position)

            # Store current position for next step
            self.agent_positions[agent_id] = current_position

    def _check_checkpoints(self, agent_id, position):
        """Check if an agent has reached any checkpoints."""
        for checkpoint in self.checkpoints:
            checkpoint_id = checkpoint["id"]
            checkpoint_pos = checkpoint["position"]
            checkpoint_radius = checkpoint["radius"]

            # Skip if agent has already passed this checkpoint
            if checkpoint_id in self.agent_checkpoints[agent_id]:
                continue

            # Calculate distance to checkpoint
            dx = position[0] - checkpoint_pos[0]
            dy = position[1] - checkpoint_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)

            # Check if agent has reached the checkpoint
            if distance <= checkpoint_radius:
                # Agent has reached this checkpoint
                self.agent_checkpoints[agent_id].add(checkpoint_id)

                # Check if this is the first agent to reach this checkpoint
                is_first = False
                if self.checkpoint_first_agent[checkpoint_id] is None:
                    self.checkpoint_first_agent[checkpoint_id] = agent_id
                    is_first = True

                # Add checkpoint info to agent's info dict in the next step
                if not hasattr(self, 'checkpoint_info'):
                    self.checkpoint_info = {}
                if agent_id not in self.checkpoint_info:
                    self.checkpoint_info[agent_id] = []

                self.checkpoint_info[agent_id].append({
                    "checkpoint_id": checkpoint_id,
                    "is_first": is_first
                })

    def _is_agent_leading(self, agent_id):
        """Check if the agent is currently leading the race."""
        if len(self.agent_progress) <= 1:
            return True  # If only one agent, it's leading by default

        # Find the agent with the most progress
        leading_agent = max(self.agent_progress.items(), key=lambda x: x[1])[0]
        return agent_id == leading_agent

    def _get_lead_margin(self, agent_id):
        """Get the margin by which an agent is leading (if it is leading)."""
        if not self._is_agent_leading(agent_id) or len(self.agent_progress) <= 1:
            return 0.0

        # Calculate lead margin (distance ahead of second place)
        progress_values = sorted(self.agent_progress.values(), reverse=True)
        if len(progress_values) > 1:
            lead_margin = progress_values[0] - progress_values[1]
            return lead_margin
        return 0.0

    def _apply_racing_rewards(self, rewards, info):
        """Apply racing-specific rewards to the base rewards."""
        # Copy rewards to avoid modifying the original
        modified_rewards = rewards.copy()

        for agent_id in self.agents.keys():
            # Add leading bonus if agent is in the lead
            if self._is_agent_leading(agent_id):
                # Base leading reward
                leading_bonus = self.config["leading_reward_factor"]

                # Additional bonus based on lead margin
                lead_margin = self._get_lead_margin(agent_id)
                margin_bonus = min(lead_margin * 0.01, 0.1)  # Cap at 0.1

                # Apply the bonuses
                modified_rewards[agent_id] += leading_bonus + margin_bonus

                # Add info about the bonuses
                if agent_id in info:
                    info[agent_id]["leading_bonus"] = leading_bonus
                    info[agent_id]["margin_bonus"] = margin_bonus

            # Add winning reward if this agent has won the race
            if self.race_winner == agent_id:
                modified_rewards[agent_id] += self.config["winning_reward"]
                if agent_id in info:
                    info[agent_id]["winning_reward"] = self.config["winning_reward"]

        return modified_rewards

    def _apply_checkpoint_rewards(self, rewards, info):
        """Apply checkpoint-specific rewards to the base rewards."""
        # Copy rewards to avoid modifying the original
        modified_rewards = rewards.copy()

        # Process checkpoint rewards for each agent
        for agent_id in self.agents.keys():
            if agent_id in self.checkpoint_info and self.checkpoint_info[agent_id]:
                # Agent has reached one or more checkpoints this step
                for checkpoint_data in self.checkpoint_info[agent_id]:
                    checkpoint_id = checkpoint_data["checkpoint_id"]
                    is_first = checkpoint_data["is_first"]

                    # Base reward for reaching a checkpoint
                    checkpoint_reward = self.config["checkpoint_reward"]

                    # Additional bonus for being first to reach this checkpoint
                    first_bonus = self.config["first_checkpoint_bonus"] if is_first else 0.0

                    # Apply the rewards
                    total_checkpoint_reward = checkpoint_reward + first_bonus
                    modified_rewards[agent_id] += total_checkpoint_reward

                    # Add info about the checkpoint rewards
                    if agent_id in info:
                        if "checkpoint_rewards" not in info[agent_id]:
                            info[agent_id]["checkpoint_rewards"] = []

                        info[agent_id]["checkpoint_rewards"].append({
                            "checkpoint_id": checkpoint_id,
                            "is_first": is_first,
                            "reward": total_checkpoint_reward
                        })

        return modified_rewards

    def _check_race_completion(self, terminated, info):
        """Check if any agent has completed the race."""
        for agent_id, agent in self.agents.items():
            # Check if agent has reached destination
            if agent_id in info and info[agent_id].get("arrive_dest", False):
                # This agent has finished the race
                if not self.race_finished:
                    self.race_finished = True
                    self.race_winner = agent_id

    def cost_function(self, vehicle_id: str):
        """
        Compute cost for safety violations.
        Implements a cost function similar to SafeMetaDriveEnv but for multi-agent racing.
        Includes incremental penalties for driving off-road.
        """
        # Calculate cost based on safety violations
        cost = 0.0
        step_info = {}

        # Get the vehicle
        vehicle = self.vehicles[vehicle_id]

        # Crash with vehicle
        if hasattr(vehicle, 'crash_vehicle') and vehicle.crash_vehicle:
            cost += self.config["crash_vehicle_cost"]
            step_info["crash_vehicle"] = True

        # Crash with object
        if hasattr(vehicle, 'crash_object') and vehicle.crash_object:
            cost += self.config["crash_object_cost"]
            step_info["crash_object"] = True

        # Out of road - with incremental penalty
        if hasattr(vehicle, 'out_of_road') and vehicle.out_of_road:
            # Initialize off-road tracking if not already done
            if not hasattr(self, 'off_road_counter'):
                self.off_road_counter = {}
            if not hasattr(self, 'off_road_status'):
                self.off_road_status = {}

            # Initialize for this vehicle if needed
            if vehicle_id not in self.off_road_counter:
                self.off_road_counter[vehicle_id] = 0
            if vehicle_id not in self.off_road_status:
                self.off_road_status[vehicle_id] = False

            # Check if this is a new off-road event or continuing
            if not self.off_road_status[vehicle_id]:
                # First time off-road
                self.off_road_status[vehicle_id] = True
                self.off_road_counter[vehicle_id] = 1
            else:
                # Continuing off-road, increment counter
                self.off_road_counter[vehicle_id] += 1

            # Apply incremental cost - increases exponentially the longer the vehicle stays off-road
            # Base cost + exponential growth based on steps off-road
            # This creates a rapidly escalating penalty to strongly incentivize returning to the road
            incremental_cost = self.config["out_of_road_cost"] * (1.0 + 0.2 * (self.off_road_counter[vehicle_id] ** 1.5))
            cost += incremental_cost

            # Add info about off-road status
            step_info["out_of_road"] = True
            step_info["off_road_steps"] = self.off_road_counter[vehicle_id]
            step_info["off_road_cost"] = incremental_cost
        else:
            # Reset off-road status when back on road
            if hasattr(self, 'off_road_status') and vehicle_id in self.off_road_status and self.off_road_status[vehicle_id]:
                self.off_road_status[vehicle_id] = False

        # Check for driving on yellow line (wrong side of road) with escalating penalty
        # Initialize wrong-side tracking if not already done
        if not hasattr(self, 'wrong_side_counter'):
            self.wrong_side_counter = {}
        if not hasattr(self, 'wrong_side_status'):
            self.wrong_side_status = {}

        # Initialize for this vehicle if needed
        if vehicle_id not in self.wrong_side_counter:
            self.wrong_side_counter[vehicle_id] = 0
        if vehicle_id not in self.wrong_side_status:
            self.wrong_side_status[vehicle_id] = False

        # Check if on yellow line (wrong side of road)
        on_wrong_side = False
        yellow_line_cost = 0.0

        if hasattr(vehicle, 'on_yellow_continuous_line') and vehicle.on_yellow_continuous_line:
            yellow_line_cost += self.config["on_yellow_line_cost"]
            step_info["on_yellow_continuous_line"] = True
            on_wrong_side = True
        if hasattr(vehicle, 'on_yellow_broken_line') and vehicle.on_yellow_broken_line:
            yellow_line_cost += self.config["on_yellow_line_cost"] * 0.5  # Less penalty for broken line
            step_info["on_yellow_broken_line"] = True
            on_wrong_side = True

        if on_wrong_side:
            # Check if this is a new wrong-side event or continuing
            if not self.wrong_side_status[vehicle_id]:
                # First time on wrong side
                self.wrong_side_status[vehicle_id] = True
                self.wrong_side_counter[vehicle_id] = 1
            else:
                # Continuing on wrong side, increment counter
                self.wrong_side_counter[vehicle_id] += 1

            # Apply exponential cost increase for staying on wrong side
            escalation_factor = 1.0 + 0.3 * (self.wrong_side_counter[vehicle_id] ** 1.5)
            yellow_line_cost *= escalation_factor

            cost += yellow_line_cost
            step_info["yellow_line_cost"] = yellow_line_cost
            step_info["wrong_side"] = True
            step_info["wrong_side_steps"] = self.wrong_side_counter[vehicle_id]
        else:
            # Reset wrong-side status when back on correct side
            if self.wrong_side_status[vehicle_id]:
                self.wrong_side_status[vehicle_id] = False

        # Check for driving on white line
        white_line_cost = 0.0
        if hasattr(vehicle, 'on_white_continuous_line') and vehicle.on_white_continuous_line:
            white_line_cost += self.config["on_white_line_cost"]
            step_info["on_white_continuous_line"] = True
        if hasattr(vehicle, 'on_white_broken_line') and vehicle.on_white_broken_line:
            white_line_cost += self.config["on_white_line_cost"] * 0.5  # Less penalty for broken line
            step_info["on_white_broken_line"] = True
        if white_line_cost > 0:
            cost += white_line_cost
            step_info["white_line_cost"] = white_line_cost

        # Track cost for this specific agent
        if vehicle_id in self.episode_cost:
            self.episode_cost[vehicle_id] += cost
            step_info["agent_cost"] = self.episode_cost[vehicle_id]
            step_info["total_cost"] = self.episode_cost[vehicle_id]

        return cost, step_info

    def done_function(self, vehicle_id):
        """
        Override the done function to allow continuous simulation.
        Only terminate when reaching the destination or max steps.
        """
        # Get the vehicle
        vehicle = self.vehicles[vehicle_id]

        # Initialize done info
        done_info = {
            "crash_vehicle": False,
            "crash_object": False,
            "out_of_road": False,
            "arrive_dest": False,
            "max_step": False
        }

        # Check for crash with vehicle
        if hasattr(vehicle, 'crash_vehicle') and vehicle.crash_vehicle:
            done_info["crash_vehicle"] = True

        # Check for crash with object
        if hasattr(vehicle, 'crash_object') and vehicle.crash_object:
            done_info["crash_object"] = True

        # Check for out of road
        if hasattr(vehicle, 'out_of_road') and vehicle.out_of_road:
            done_info["out_of_road"] = True

        # Check for arrival at destination
        if hasattr(vehicle, 'arrive_destination') and vehicle.arrive_destination:
            done_info["arrive_dest"] = True

        # Check for max steps
        if self.episode_step >= self.config["horizon"]:
            done_info["max_step"] = True

        # Only terminate for max steps or arrival at destination
        done = done_info["max_step"] or done_info["arrive_dest"]

        return done, done_info


if __name__ == "__main__":
    # Example usage
    env = MultiAgentRacingSafeEnv(
        {
            "use_render": True,
            "manual_control": True,
            "num_agents": 2,
            "map": "CSRCR",  # Use a circular map for racing
            "vehicle_config": {
                "enable_reverse": True,
                "show_lidar": False,
                "show_side_detector": False,
                "show_lane_line_detector": False,
            }
        }
    )

    # Reset the environment
    obs, _ = env.reset()

    # Main simulation loop
    for i in range(1, 100000):
        # Random actions for demonstration
        actions = {agent_id: [0, 0] for agent_id in env.agents.keys()}

        # Step the environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Render with race information
        env.render(
            text={
                "Race Progress": {agent_id: f"{progress:.2f}" for agent_id, progress in env.agent_progress.items()},
                "Leading": next((agent_id for agent_id in env.agents.keys() if env.agent_progress.get(agent_id, 0) == max(env.agent_progress.values())), None),
                "Race Winner": env.race_winner if env.race_finished else "None",
            }
        )

        # Check if all agents are done
        if all(terminated.values()):
            print("All agents terminated. Resetting environment.")
            obs, _ = env.reset()

    env.close()
