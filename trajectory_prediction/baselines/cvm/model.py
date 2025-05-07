import numpy as np
import torch


def predict_trajectory(trajectory: np.ndarray, sample=True) -> np.ndarray:
    """Generate 20 diverse possible future trajectories using constant velocity model with stochastic sampling.

    Args:
        trajectory (np.ndarray): Input trajectory of shape [num_agents, traj_length, 2], where traj_length is 8.

    Returns:
        np.ndarray: An array of 20 diverse trajectories of shape [20, num_agents, 12, 2].
    """
    # Convert numpy array to torch tensor for compatibility with reference code
    trajectory_torch = torch.tensor(trajectory, dtype=torch.float32)
    num_agents = trajectory.shape[0]

    # Parameters from the reference code
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

            # Calculate relative movement (similar to obs_rel in the reference)
            obs_rel = observed[:, 1:] - observed[:, :-1]  # [1, 7, 2]

            # Use the last velocity as constant velocity
            last_velocity = obs_rel[:, -1].unsqueeze(1)  # [1, 1, 2]

            # Apply rotation to the velocity (using the same rotation for all agents)
            vel = last_velocity.squeeze(dim=0).squeeze(dim=0)  # [2]
            rotated_vel = torch.matmul(rotation_mat, vel)  # [2]
            rotated_vel = rotated_vel.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]

            # Create 12 predictions with the constant (rotated) velocity
            pred_rel = rotated_vel.repeat(1, 12, 1)  # [1, 12, 2]

            # Convert to absolute positions (cumulative sum + start position)
            displacement = torch.cumsum(pred_rel, dim=1)  # [1, 12, 2]
            start_pos = observed[:, -1].unsqueeze(1)  # [1, 1, 2]
            pred_abs = displacement + start_pos  # [1, 12, 2]

            predictions.append(pred_abs.squeeze(0).numpy())  # [12, 2]

        # Stack all agent predictions for this sample
        sample_prediction = np.stack(predictions, axis=0)  # [num_agents, 12, 2]
        all_trajectories.append(sample_prediction)

    # Stack all samples
    result = np.stack(all_trajectories, axis=0)  # [20, num_agents, 12, 2]
    return result
