#!/usr/bin/env python
"""
This script demonstrates how to use the Multi-Agent Racing Safe MetaDrive environment.

Key features:
1. Two cars racing against each other
2. Modified reward structure to incentivize racing
3. Continuous simulation despite collisions or boundary violations
4. Support for reverse driving
5. Both agents consistently use their RL models regardless of camera focus

Note: The policy classes have been modified to ensure that both agents use their
reinforcement learning (RL) models at all times, regardless of which car is currently
being focused on by the viewer. This ensures that when switching between agents with
the camera (using the Q key), the non-focused agent continues to use its RL model
rather than reverting to a simple straight-line behavior.

Please feel free to run this script to enjoy a racing experience! Remember to press H to see help message!
"""
import argparse

from metadrive.constants import HELP_MESSAGE
from metadrive.envs.marl_safe_metadrive_env import MultiAgentRacingSafeEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the environment")
    parser.add_argument("--manual_control", action="store_true", help="Enable manual control for testing")
    args = parser.parse_args()

    # Configure environment
    env = MultiAgentRacingSafeEnv(
        {
            "use_render": True,
            "manual_control": args.manual_control,
            "num_agents": 2,
            "start_seed": args.seed,
            "num_scenarios": 1,  # Use a single scenario for racing
            "map": "CSRCR",  # Use a circular map for racing
            # Enable AI protector to ensure both agents use autodrive
            "use_AI_protector": True,
            "vehicle_config": {
                "enable_reverse": True,
                "show_lidar": False,
                "show_side_detector": False,
                "show_lane_line_detector": False,
            }
        }
    )

    # Print help message
    print(HELP_MESSAGE)
    print("\nAdditional Racing Environment Controls:")
    print("- Press Q to toggle between agents (when manual_control is enabled)")
    print("- Press R to enable/disable reverse mode")
    print("- Press T to toggle between manual control and autodrive for the current agent")
    print("- Both agents use autodrive by default and will continue driving even when not focused")
    print("- Collisions and out-of-road events will not terminate the episode")
    print("- The episode will only end when reaching the destination or max steps")
    print("- The leading car receives bonus rewards")
    print("- The first car to complete the track receives a winning reward")

    # Reset the environment
    obs, _ = env.reset()

    # Set expert_takeover to True for all agents to enable autodrive
    for agent_id, agent in env.agents.items():
        agent.expert_takeover = True
        print(f"Set agent {agent_id} to expert takeover mode")

    # Main simulation loop
    for i in range(1, 100000):
        # Default action is to do nothing
        actions = {agent_id: [0, 0] for agent_id in env.agents.keys()}

        # Step the environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Get race information
        leading_agent = next((agent_id for agent_id in env.agents.keys()
                             if env.agent_progress.get(agent_id, 0) == max(env.agent_progress.values())), None)

        # Extract current step costs from info
        step_costs = {}
        for agent_id, agent_info in info.items():
            # Get the current step cost if available
            if "cost" in agent_info:
                step_costs[agent_id] = agent_info["cost"]
            else:
                step_costs[agent_id] = 0.0

        # Render with race information
        env.render(
            text={
                "Race Progress": {agent_id: f"{progress:.2f}" for agent_id, progress in env.agent_progress.items()},
                "Leading": leading_agent,
                "Race Winner": env.race_winner if env.race_finished else "None",
                "Rewards": {agent_id: f"{reward:.2f}" for agent_id, reward in rewards.items()},
                "Step Cost": {agent_id: f"{step_costs.get(agent_id, 0):.2f}" for agent_id in env.agents.keys()},
                "Total Cost": {agent_id: f"{env.episode_cost.get(agent_id, 0):.2f}" for agent_id in env.agents.keys()},
                "Off-Road": {agent_id: f"Yes ({env.off_road_counter.get(agent_id, 0)} steps)" if env.off_road_status.get(agent_id, False) else "No" for agent_id in env.agents.keys()},
                "Wrong Side": {agent_id: f"Yes ({env.wrong_side_counter.get(agent_id, 0)} steps)" if env.wrong_side_status.get(agent_id, False) else "No" for agent_id in env.agents.keys()},
                "On Line": {agent_id: "Yellow" if info.get(agent_id, {}).get("on_yellow_continuous_line", False) or info.get(agent_id, {}).get("on_yellow_broken_line", False) else "White" if info.get(agent_id, {}).get("on_white_continuous_line", False) or info.get(agent_id, {}).get("on_white_broken_line", False) else "No" for agent_id in env.agents.keys()},
                "Checkpoints": {agent_id: f"{info.get(agent_id, {}).get('checkpoints_passed', 0)}/{info.get(agent_id, {}).get('total_checkpoints', 0)}" for agent_id in env.agents.keys()},
            }
        )

        # Check if all agents are done
        if all(terminated.values()):
            print("All agents terminated. Resetting environment.")
            obs, _ = env.reset()

    env.close()
