import numpy as np
import torch


def predict_trajectory(trajectory: np.ndarray, sample=True) -> np.ndarray:
    """
    Generate 20 diverse possible future trajectories using Constant Turn Rate and Velocity (CTRV) model.

    Args:
        trajectory (np.ndarray): [num_agents, traj_length, 2], traj_length = 8
        sample (bool): Whether to add angular noise to predictions.

    Returns:
        np.ndarray: [20, num_agents, 12, 2] predicted future trajectories
    """
    trajectory_torch = torch.tensor(trajectory, dtype=torch.float32)
    num_agents = trajectory.shape[0]

    # Parameters
    num_samples = 20
    pred_steps = 12
    sample_angle_std = 25  # degrees
    dt = 1

    # Calculate velocities and headings from last 3 positions
    last_pos = trajectory_torch[:, -1]  # [num_agents, 2]
    prev_pos = trajectory_torch[:, -2]  # [num_agents, 2]
    prev_prev_pos = trajectory_torch[:, -3]  # [num_agents, 2]

    # Calculate velocities
    velocity_vec = (last_pos - prev_pos) / dt  # [num_agents, 2]
    prev_velocity_vec = (prev_pos - prev_prev_pos) / dt  # [num_agents, 2]

    # Calculate speeds
    speed = torch.norm(velocity_vec, dim=1, keepdim=True)  # [num_agents, 1]

    # Calculate headings
    heading = torch.atan2(velocity_vec[:, 1], velocity_vec[:, 0])  # [num_agents]
    prev_heading = torch.atan2(
        prev_velocity_vec[:, 1], prev_velocity_vec[:, 0]
    )  # [num_agents]

    # Calculate turn rates
    turn_rate = (heading - prev_heading) / dt  # [num_agents]

    all_trajectories = []

    for _ in range(num_samples):
        if sample:
            sampled_angle = np.random.normal(0, sample_angle_std, 1)[0]
        else:
            sampled_angle = 0.0

        # Convert degrees to radians
        theta = (sampled_angle * np.pi) / 180.0

        # Initialize trajectory container
        future_positions = torch.zeros((num_agents, pred_steps, 2), dtype=torch.float32)
        pos = last_pos.clone()
        current_heading = heading.clone()
        current_turn_rate = turn_rate.clone()
        current_speed = speed.clone()

        for t in range(pred_steps):
            # For each agent
            for agent_idx in range(num_agents):
                # If turn rate is close to zero, use linear motion
                if torch.abs(current_turn_rate[agent_idx]) < 1e-6:
                    dx = (
                        current_speed[agent_idx]
                        * torch.cos(current_heading[agent_idx])
                        * dt
                    )
                    dy = (
                        current_speed[agent_idx]
                        * torch.sin(current_heading[agent_idx])
                        * dt
                    )
                else:
                    # Use CTRV motion model
                    dx = (current_speed[agent_idx] / current_turn_rate[agent_idx]) * (
                        torch.sin(
                            current_heading[agent_idx] + current_turn_rate[agent_idx] * dt
                        )
                        - torch.sin(current_heading[agent_idx])
                    )
                    dy = (current_speed[agent_idx] / current_turn_rate[agent_idx]) * (
                        -torch.cos(
                            current_heading[agent_idx] + current_turn_rate[agent_idx] * dt
                        )
                        + torch.cos(current_heading[agent_idx])
                    )

                # Update position
                pos[agent_idx] += torch.tensor([dx, dy])
                # Update heading
                current_heading[agent_idx] += current_turn_rate[agent_idx] * dt

                # Store position
                future_positions[agent_idx, t] = pos[agent_idx]

        all_trajectories.append(future_positions)

    # Stack all samples -> [20, num_agents, 12, 2]
    all_trajectories = torch.stack(all_trajectories, dim=0)
    return all_trajectories.numpy()
