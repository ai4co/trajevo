import numpy as np


def predict_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Generate 20 possible future trajectories, focusing on top heuristics and refinement.

    Args:
        trajectory: [num_agents, traj_length, 2] past trajectory (traj_length = 8)

    Returns:
        20 diverse trajectories [20, num_agents, 12, 2]
    """
    num_agents = trajectory.shape[0]
    all_trajectories = []

    # Helper function to calculate velocity
    def calculate_velocity(traj):
        return np.median(traj[:, -3:, :] - traj[:, -4:-1, :], axis=1)

    # 1. Adaptive Damped Momentum (Refined) - HIGH PRIORITY
    velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
    agent_speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    damping = np.clip(0.93 - agent_speed * 0.012, 0.85, 0.98)
    current_pos = trajectory[:, -1, :]
    predictions = []
    noise_scale = 0.00005
    for t in range(12):
        velocity = velocity * damping + np.random.normal(
            0, noise_scale, size=(num_agents, 2)
        )
        current_pos = current_pos + velocity
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)  # Index 0

    # 2. Adaptive Damping based on agent speed (Enhanced) - HIGH PRIORITY
    velocity = calculate_velocity(trajectory)
    agent_speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    damping = np.clip(1 - agent_speed * 0.009, 0.87, 0.99)  # Slower = more damping
    current_pos = trajectory[:, -1, :]
    predictions = []
    velocity_curr = trajectory[:, -1, :] - trajectory[:, -2, :]
    noise_scale = 0.00004
    for t in range(12):
        velocity_curr = velocity_curr * damping + np.random.normal(
            0, noise_scale, size=(num_agents, 2)
        )
        current_pos = current_pos + velocity_curr
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)  # Index 1

    # 3. Smoothed Constant Velocity (Focus on this with variation) - Smoother
    velocity = (trajectory[:, -1, :] - trajectory[:, 0, :]) / 7.0
    smoothed_velocity = velocity + np.random.normal(0, 0.00015, size=(num_agents, 2))
    smoothed_traj = trajectory[:, -1:, :] + np.expand_dims(
        np.arange(1, 13), axis=(0, 2)
    ) * np.expand_dims(smoothed_velocity, axis=1)
    all_trajectories.append(smoothed_traj)  # Index 2

    # 4. Slight Acceleration (Adaptive Noise)
    velocity = calculate_velocity(trajectory)
    acceleration = (trajectory[:, -1, :] - trajectory[:, -2, :]) - (
        trajectory[:, -2, :] - trajectory[:, -3, :]
    )
    acceleration_scale = np.linalg.norm(acceleration)
    noise_scale = 0.00005 + acceleration_scale * 0.0002
    current_pos = trajectory[:, -1, :]
    predictions = []
    for t in range(12):
        noise = np.random.normal(0, noise_scale, size=(num_agents, 2))
        current_pos = current_pos + velocity + (acceleration + noise) * (t + 1) * 0.25
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)  # Index 3

    # 5. Circumvent Obstacle (Moderate Rotation, higher prob)
    velocity = calculate_velocity(trajectory)
    angle = np.random.uniform(-np.pi / 45, np.pi / 45)
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated_velocity = np.dot(velocity, rotation_matrix)
    circumvent_trajectory = trajectory[:, -1:, :] + np.expand_dims(
        np.arange(1, 13), axis=(0, 2)
    ) * np.expand_dims(rotated_velocity, axis=1)
    all_trajectories.append(circumvent_trajectory)  # Index 4

    # 6. Adaptive Damped Momentum - Varying Damping
    velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
    agent_speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    damping = np.clip(0.90 - agent_speed * 0.010, 0.82, 0.95)  # Wider range
    current_pos = trajectory[:, -1, :]
    predictions = []
    noise_scale = 0.00006  # Slightly higher noise
    for t in range(12):
        velocity = velocity * damping + np.random.normal(
            0, noise_scale, size=(num_agents, 2)
        )
        current_pos = current_pos + velocity
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)

    # 7. Smoothed with current velocity as a Momentum Correction
    velocity = (trajectory[:, -1, :] - trajectory[:, 0, :]) / 7.0
    current_velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
    corrected_velocity = (
        0.8 * velocity
        + 0.2 * current_velocity
        + np.random.normal(0, 0.0001, size=(num_agents, 2))
    )
    smoothed_traj = trajectory[:, -1:, :] + np.expand_dims(
        np.arange(1, 13), axis=(0, 2)
    ) * np.expand_dims(corrected_velocity, axis=1)
    all_trajectories.append(smoothed_traj)

    # 8. Combination: Damped Momentum and Smoothed (More Weight on Damped)
    damped = all_trajectories[0]
    smoothed = all_trajectories[2]
    combined_ds = (
        0.75 * damped + 0.25 * smoothed + np.random.normal(0, 0.00025, size=damped.shape)
    )
    all_trajectories.append(combined_ds)  # Index 5

    # 9. More Randomness and Drift (Slightly Increased Drift)
    drift_scale = 0.0008
    current_pos = trajectory[:, -1, :]
    predictions = []
    for t in range(12):
        drift = np.random.normal(0, drift_scale, size=(num_agents, 2))
        current_pos = current_pos + drift
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)  # Index 6

    # 10. Adaptive Damping with slightly increased speed
    velocity = calculate_velocity(trajectory) * 1.03
    agent_speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    damping = np.clip(1 - agent_speed * 0.01, 0.86, 0.98)
    current_pos = trajectory[:, -1, :]
    predictions = []
    velocity_curr = trajectory[:, -1, :] - trajectory[:, -2, :]
    noise_scale = 0.00002
    for t in range(12):
        velocity_curr = velocity_curr * damping + np.random.normal(
            0, noise_scale, size=(num_agents, 2)
        )
        current_pos = current_pos + velocity_curr
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)  # Index 7

    # 11. Smoothed Constant Velocity with a look back of 4 frames
    velocity = (trajectory[:, -1, :] - trajectory[:, 3, :]) / 4.0
    smoothed_velocity = velocity + np.random.normal(0, 0.0001, size=(num_agents, 2))
    smoothed_traj = trajectory[:, -1:, :] + np.expand_dims(
        np.arange(1, 13), axis=(0, 2)
    ) * np.expand_dims(smoothed_velocity, axis=1)
    all_trajectories.append(smoothed_traj)

    # 12. Combined Damped Momentum with Slight Acceleration
    damped = all_trajectories[0]
    acceleration = all_trajectories[3]
    combined_da = (
        0.7 * damped + 0.3 * acceleration + np.random.normal(0, 0.0001, size=damped.shape)
    )
    all_trajectories.append(combined_da)

    # 13. Circumvent with more aggressive angle change
    velocity = calculate_velocity(trajectory)
    angle = np.random.uniform(-np.pi / 30, np.pi / 30)
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated_velocity = np.dot(velocity, rotation_matrix)
    circumvent_trajectory = trajectory[:, -1:, :] + np.expand_dims(
        np.arange(1, 13), axis=(0, 2)
    ) * np.expand_dims(rotated_velocity, axis=1)
    all_trajectories.append(circumvent_trajectory)

    # 14. Adaptive damping using a different velocity calculation window.
    velocity = np.median(trajectory[:, -5:, :] - trajectory[:, -6:-1, :], axis=1)
    agent_speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    damping = np.clip(1 - agent_speed * 0.009, 0.87, 0.99)  # Slower = more damping
    current_pos = trajectory[:, -1, :]
    predictions = []
    velocity_curr = trajectory[:, -1, :] - trajectory[:, -2, :]
    noise_scale = 0.00004
    for t in range(12):
        velocity_curr = velocity_curr * damping + np.random.normal(
            0, noise_scale, size=(num_agents, 2)
        )
        current_pos = current_pos + velocity_curr
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)

    # 15. Combined: Top 3 (Damped, Adaptive, Smoothed) - adjusted weights
    damped = all_trajectories[0]
    adaptive = all_trajectories[1]
    smoothed = all_trajectories[2]
    combined_das = (0.45 * damped + 0.35 * adaptive + 0.2 * smoothed) + np.random.normal(
        0, 0.00015, size=damped.shape
    )
    all_trajectories.append(combined_das)

    # 16. Damped momentum with speed scaling factor.
    velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
    agent_speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    velocity = velocity * np.clip(1 + agent_speed * 0.1, 0.95, 1.05)
    damping = np.clip(0.93 - agent_speed * 0.012, 0.85, 0.98)
    current_pos = trajectory[:, -1, :]
    predictions = []
    noise_scale = 0.00005
    for t in range(12):
        velocity = velocity * damping + np.random.normal(
            0, noise_scale, size=(num_agents, 2)
        )
        current_pos = current_pos + velocity
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)

    # Fill remaining with variations of best trajectories
    while len(all_trajectories) < 20:
        idx1 = np.random.randint(0, min(len(all_trajectories), 7))  # Prioritize top 7
        idx2 = np.random.randint(0, len(all_trajectories))
        traj1 = all_trajectories[idx1]
        traj2 = all_trajectories[idx2]
        avg_traj = (0.65 * traj1 + 0.35 * traj2) + np.random.normal(
            0, 0.00005, size=traj1.shape
        )
        all_trajectories.append(avg_traj)

    all_trajectories = np.stack(all_trajectories, axis=0)
    return all_trajectories
