import numpy as np


def predict_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Generate 20 possible future trajectories with improved heuristics,
    focusing on linear, accelerated and turning motion, with refined parameters and enhanced diversity.

    Args:
        trajectory (np.ndarray): [num_agents, traj_length, 2] - Past trajectory (traj_length=8).

    Returns:
        np.ndarray: [20, num_agents, 12, 2] - 20 diverse future trajectories.
    """

    num_agents = trajectory.shape[0]
    all_trajectories = []

    # Precompute velocity using a longer window for stability
    velocity = np.mean(
        trajectory[:, -4:, :] - trajectory[:, -5:-1, :], axis=1, keepdims=False
    )
    acceleration = np.mean(
        velocity[:, :] - velocity[:, :], axis=1, keepdims=False
    )  # Approximation of acceleration

    for i in range(20):
        current_pos = trajectory[:, -1, :]

        # Trajectories 0-5: Linear Extrapolation with refined noise and velocity averaging
        if i < 6:
            avg_window = 2 + (i % 4)  # Vary averaging window between 2 and 5
            velocity_current = np.mean(
                trajectory[:, -avg_window:, :] - trajectory[:, -(avg_window + 1) : -1, :],
                axis=1,
            )
            noise_scale = np.random.uniform(0.00002, 0.00008)  # Varying noise scale
            predictions = []
            for t in range(1, 13):
                noise = np.random.normal(
                    0, noise_scale * (t**0.4), size=current_pos.shape
                )
                current_pos = current_pos + velocity_current + noise * 0.25
                predictions.append(current_pos.copy())
            pred_trajectory = np.stack(predictions, axis=1)

        # Trajectories 6-9: Damping Motion with varying damping factors and slight noise
        elif i < 10:
            damping_factor = np.random.uniform(0.85, 0.99)  # Damping factor range
            velocity_current = trajectory[:, -1, :] - trajectory[:, -2, :]
            noise_scale = np.random.uniform(0.000004, 0.000016)
            predictions = []
            for t in range(1, 13):
                noise = np.random.normal(0, noise_scale, size=current_pos.shape)
                current_pos = current_pos + velocity_current + noise * 0.035
                velocity_current *= damping_factor
                predictions.append(current_pos.copy())
            pred_trajectory = np.stack(predictions, axis=1)

        # Trajectories 10-13: Slight Turning with slight acceleration
        elif i < 14:
            turn_factor = np.random.uniform(-0.020, 0.020)  # Turning factor range
            acceleration_factor = np.random.uniform(0.00001, 0.00006)
            velocity_current = trajectory[:, -1, :] - trajectory[:, -2, :]
            rotation_matrix = np.array(
                [
                    [np.cos(turn_factor), -np.sin(turn_factor)],
                    [np.sin(turn_factor), np.cos(turn_factor)],
                ]
            )
            predictions = []
            noise_scale = np.random.uniform(0.000004, 0.000010)
            for t in range(1, 13):
                noise = np.random.normal(0, noise_scale, size=current_pos.shape)
                velocity_current = np.einsum(
                    "ij,kj->ki", rotation_matrix, velocity_current
                )  # Rotate velocities
                velocity_current += acceleration_factor
                current_pos = current_pos + velocity_current + noise * 0.025
                predictions.append(current_pos.copy())
            pred_trajectory = np.stack(predictions, axis=1)

        # Trajectories 14-19: More Diverse Turning Motion and variable damping
        else:
            turn_factor = np.random.uniform(-0.04, 0.06)  # turning from -0.04 to 0.06
            damping = np.random.uniform(0.992, 0.998)  # damping
            noise_scale = np.random.uniform(0.000005, 0.000012)  # noise
            predictions = []
            velocity_current = velocity.copy()
            rotation_matrix = np.array(
                [
                    [np.cos(turn_factor), -np.sin(turn_factor)],
                    [np.sin(turn_factor), np.cos(turn_factor)],
                ]
            )

            for t in range(1, 13):
                noise = np.random.normal(
                    0, noise_scale * (t**0.3), size=current_pos.shape
                )
                velocity_current = np.einsum(
                    "ij,aj->ai", rotation_matrix, velocity_current
                )
                velocity_current *= damping
                current_pos = current_pos + velocity_current + noise * 0.03
                predictions.append(current_pos.copy())
            pred_trajectory = np.stack(predictions, axis=1)

        all_trajectories.append(pred_trajectory)

    all_trajectories = np.stack(all_trajectories, axis=0)

    return all_trajectories
