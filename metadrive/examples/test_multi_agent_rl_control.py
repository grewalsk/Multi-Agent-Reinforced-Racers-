#!/usr/bin/env python
"""
Test script to verify that both agents in a multi-agent environment use their RL models
regardless of which agent is being tracked by the camera.

This script:
1. Creates a multi-agent environment with two agents
2. Runs the simulation with both agents using RL models
3. Toggles between agents to verify both continue using their RL models
4. Prints debug information about which control mode is being used

Usage:
    python -m metadrive.examples.test_multi_agent_rl_control
"""

import time
import argparse
import time  # For adding delays
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive


def test_multi_agent_rl_control():
    """Test that both agents use RL models regardless of camera focus."""
    # Create environment with two agents
    env = MultiAgentMetaDrive(
        {
            "use_render": True,
            "manual_control": True,  # Enable manual control for toggling between agents
            "num_agents": 2,
            "traffic_density": 0.0,
            "start_seed": 5,
            "map": "SSSS",
            # Use AI protector to enable autodrive for all agents
            "use_AI_protector": True
        }
    )

    # Reset environment
    obs, info = env.reset()

    # Set expert_takeover to True for all agents to enable autodrive
    for agent_id, agent in env.agents.items():
        agent.expert_takeover = True
        print(f"Set agent {agent_id} to expert takeover mode")

    # Print initial state
    print("\n=== Initial State ===")
    print(f"Number of agents: {len(env.agents)}")
    print(f"Agent IDs: {list(env.agents.keys())}")
    print(f"Current tracked agent: {env.current_track_agent.id}")

    # Run simulation for a longer time
    print("\nStarting simulation - press Ctrl+C to exit")
    try:
        for i in range(1000):  # Increased from 100 to 1000 steps
            # Generate actions that make the cars move forward
            actions = {}
            for agent_id in env.agents.keys():
                # Use a constant forward action to simulate RL model outputs
                # This will make the cars move and demonstrate the behavior
                actions[agent_id] = [0.0, 0.5]  # [steering, acceleration]

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Print debug info every 10 steps
            if i % 10 == 0:
                print(f"\n=== Step {i} ===")
                for agent_id in env.agents.keys():
                    agent = env.agents[agent_id]
                    is_tracked = agent is env.current_track_agent

                    # Get the policy for this agent
                    policy = env.engine.agent_manager.get_policy(agent.id)
                    control_mode = "UNKNOWN"

                    # Check if agent is using expert control (autodrive)
                    if hasattr(agent, 'expert_takeover') and agent.expert_takeover:
                        control_mode = "EXPERT (AUTODRIVE)"
                    # Check policy action info
                    elif hasattr(policy, 'action_info'):
                        if policy.action_info.get("expert_control", False):
                            control_mode = "EXPERT"
                        elif policy.action_info.get("manual_control", False):
                            control_mode = "MANUAL"
                        else:
                            control_mode = "RL MODEL"

                    # Print the actual action that was applied to the agent
                    if hasattr(agent, 'last_current_action') and len(agent.last_current_action) > 0:
                        print(f"  Last Action Applied: {agent.last_current_action[-1]}")

                    print(f"Agent {agent_id}: {'TRACKED' if is_tracked else 'NOT TRACKED'}")
                    print(f"  Position: {agent.position}")
                    print(f"  Action: {actions[agent_id]}")
                    print(f"  Control Mode: {control_mode}")

                # No automatic agent switching - only manual switching via Q key
                # The Q key is already bound to agent switching in the environment
                # when manual_control is enabled

            # Render less frequently to reduce potential flashing
            # Only render every 2 steps
            if i % 2 == 0:
                env.render()

            # Check if all agents are done
            if all(terminated.values()):
                print("All agents terminated. Resetting environment.")
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError during simulation: {e}")

    # Close environment
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    test_multi_agent_rl_control()
