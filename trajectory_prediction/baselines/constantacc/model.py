import numpy as np
import torch


def predict_trajectory(trajectory: np.ndarray, sample=True) -> np.ndarray:
    """Generate 20 diverse possible future trajectories using constant acceleration model with stochastic sampling.

    Args:
        trajectory (np.ndarray): Input trajectory of shape [num_agents, traj_length, 2], where traj_length is 8.
        sample (bool): Whether to add angular noise to predictions. Defaults to True.

    Returns:
        np.ndarray: An array of 20 diverse trajectories of shape [20, num_agents, 12, 2].
    """
    # Convert numpy array to torch tensor for compatibility
    trajectory_torch = torch.tensor(trajectory, dtype=torch.float32)
    num_agents = trajectory.shape[0]

    # Parameters
    num_samples = 20
    sample_angle_std = 25  # Standard deviation for angle sampling in degrees

    all_trajectories = []

    # For each sample
    for _ in range(num_samples):
        # Sample one angle for all agents in this trajectory sample
        if sample:
            sampled_angle = np.random.normal(0, sample_angle_std, 1)[0]
        else:
            sampled_angle = 0

        theta = (sampled_angle * np.pi) / 180.0
        c, s = np.cos(theta), np.sin(theta)
        rotation_mat = torch.tensor([[c, s], [-s, c]], dtype=torch.float32)

        predictions = []

        for agent_idx in range(num_agents):
            # Get observed trajectory for current agent
            observed = trajectory_torch[agent_idx].unsqueeze(0)  # [1, 8, 2]

            # Calculate velocities and acceleration
            # Get last three positions
            last_three_pos = observed[:, -3:]  # [1, 3, 2]

            # Calculate velocities between consecutive positions
            velocities = last_three_pos[:, 1:] - last_three_pos[:, :-1]  # [1, 2, 2]

            # Calculate acceleration (difference between consecutive velocities)
            acceleration = velocities[:, 1:] - velocities[:, :-1]  # [1, 1, 2]

            # Get last velocity
            last_velocity = velocities[:, -1].unsqueeze(1)  # [1, 1, 2]

            # Apply rotation to both velocity and acceleration
            vel = last_velocity.squeeze(dim=0).squeeze(dim=0)  # [2]
            acc = acceleration.squeeze(dim=0).squeeze(dim=0)  # [2]

            rotated_vel = torch.matmul(rotation_mat, vel)  # [2]
            rotated_acc = torch.matmul(rotation_mat, acc)  # [2]

            rotated_vel = rotated_vel.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
            rotated_acc = rotated_acc.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]

            # Create predictions with constant acceleration
            time_steps = torch.arange(1, 13, dtype=torch.float32).view(
                1, -1, 1
            )  # [1, 12, 1]

            # Position = initial_position + velocity * t + 0.5 * acceleration * t^2
            displacement = rotated_vel * time_steps + 0.5 * rotated_acc * (
                time_steps**2
            )  # [1, 12, 2]

            start_pos = observed[:, -1].unsqueeze(1)  # [1, 1, 2]
            pred_abs = displacement + start_pos  # [1, 12, 2]

            predictions.append(pred_abs.squeeze(0).numpy())  # [12, 2]

        # Stack all agent predictions for this sample
        sample_prediction = np.stack(predictions, axis=0)  # [num_agents, 12, 2]
        all_trajectories.append(sample_prediction)

    # Stack all samples
    result = np.stack(all_trajectories, axis=0)  # [20, num_agents, 12, 2]
    return result
