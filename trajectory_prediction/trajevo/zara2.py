import numpy as np


def predict_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """
    Generate 20 possible future trajectories, focusing on top heuristics and refinement.

    Args:
        trajectory: [num_agents, traj_length, 2] past trajectory (traj_length = 8)

    Returns:
        20 diverse trajectories [20, num_agents, 12, 2]
    """
    num_agents = trajectory.shape[0]
    all_trajectories = []
    future_len = 12
    time_steps = np.arange(1, future_len + 1).reshape(1, future_len, 1)

    def calculate_velocity(traj):
        return np.median(traj[:, -3:, :] - traj[:, -4:-1, :], axis=1)

    # 0. Goal-Directed Curve Model
    current_pos = trajectory[:, -1, :].copy()
    long_term_velocity = (trajectory[:, -1, :] - trajectory[:, 0, :]) / 7.0
    goal_point = current_pos + long_term_velocity * future_len
    current_vel = calculate_velocity(trajectory)
    predictions = []
    turn_strength = 0.2
    for _ in range(future_len):
        direction_to_goal = goal_point - current_pos
        direction_to_goal_norm = direction_to_goal / (
            np.linalg.norm(direction_to_goal, axis=1, keepdims=True) + 1e-6
        )
        current_vel_norm = current_vel / (
            np.linalg.norm(current_vel, axis=1, keepdims=True) + 1e-6
        )
        new_direction = (
            1 - turn_strength
        ) * current_vel_norm + turn_strength * direction_to_goal_norm
        speed = np.linalg.norm(current_vel, axis=1, keepdims=True)
        current_vel = new_direction * speed
        current_pos = current_pos + current_vel
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)

    # 1. Adaptive Damping based on agent speed (Enhanced)
    velocity = calculate_velocity(trajectory)
    agent_speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    damping = np.clip(1 - agent_speed * 0.009, 0.87, 0.99)
    current_pos = trajectory[:, -1, :]
    predictions = []
    velocity_curr = trajectory[:, -1, :] - trajectory[:, -2, :]
    noise_scale = 0.00004
    for t in range(future_len):
        velocity_curr = velocity_curr * damping + np.random.normal(
            0, noise_scale, size=(num_agents, 2)
        )
        current_pos = current_pos + velocity_curr
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)

    # 2. Smoothed Constant Velocity
    velocity = (trajectory[:, -1, :] - trajectory[:, 0, :]) / 7.0
    smoothed_velocity = velocity + np.random.normal(0, 0.00015, size=(num_agents, 2))
    smoothed_traj = trajectory[:, -1:, :] + time_steps * np.expand_dims(
        smoothed_velocity, axis=1
    )
    all_trajectories.append(smoothed_traj)

    # 3. Slight Acceleration (The "jerky" but effective original model)
    velocity = calculate_velocity(trajectory)
    acceleration = (trajectory[:, -1, :] - trajectory[:, -2, :]) - (
        trajectory[:, -2, :] - trajectory[:, -3, :]
    )
    acceleration_scale = np.linalg.norm(acceleration)
    noise_scale = 0.00005 + acceleration_scale * 0.0002
    current_pos = trajectory[:, -1, :]
    predictions = []
    for t in range(future_len):
        noise = np.random.normal(0, noise_scale, size=(num_agents, 2))
        current_pos = current_pos + velocity + (acceleration + noise) * (t + 1) * 0.25
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)

    # 4. Kinematic model using the winning "jerky" acceleration
    initial_pos = trajectory[:, -1:, :]
    initial_vel = calculate_velocity(trajectory)[:, np.newaxis, :]
    jerky_accel = (
        (trajectory[:, -1, :] - trajectory[:, -2, :])
        - (trajectory[:, -2, :] - trajectory[:, -3, :])
    )[:, np.newaxis, :]
    kinematic_traj = (
        initial_pos + initial_vel * time_steps + 0.5 * jerky_accel * (time_steps**2)
    )
    all_trajectories.append(kinematic_traj)

    # 5. Adaptive Damped Momentum - Varying Damping
    velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
    agent_speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    damping = np.clip(0.90 - agent_speed * 0.010, 0.82, 0.95)
    current_pos = trajectory[:, -1, :]
    predictions = []
    noise_scale = 0.00006
    for t in range(future_len):
        velocity = velocity * damping + np.random.normal(
            0, noise_scale, size=(num_agents, 2)
        )
        current_pos = current_pos + velocity
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)

    # 6. Smoothed with current velocity as a Momentum Correction
    velocity = (trajectory[:, -1, :] - trajectory[:, 0, :]) / 7.0
    current_velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
    corrected_velocity = (
        0.8 * velocity
        + 0.2 * current_velocity
        + np.random.normal(0, 0.0001, size=(num_agents, 2))
    )
    smoothed_traj = trajectory[:, -1:, :] + time_steps * np.expand_dims(
        corrected_velocity, axis=1
    )
    all_trajectories.append(smoothed_traj)

    # 7. Combination: Damped Momentum and Smoothed
    damped = all_trajectories[1]
    smoothed = all_trajectories[2]
    combined_ds = (
        0.75 * damped + 0.25 * smoothed + np.random.normal(0, 0.00025, size=damped.shape)
    )
    all_trajectories.append(combined_ds)

    # 8. More Randomness and Drift
    drift_scale = 0.0008
    current_pos = trajectory[:, -1, :]
    predictions = []
    for t in range(future_len):
        drift = np.random.normal(0, drift_scale, size=(num_agents, 2))
        current_pos = current_pos + drift
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)

    # 9. Adaptive Damping with slightly increased speed
    velocity = calculate_velocity(trajectory) * 1.03
    agent_speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    damping = np.clip(1 - agent_speed * 0.01, 0.86, 0.98)
    current_pos = trajectory[:, -1, :]
    predictions = []
    velocity_curr = trajectory[:, -1, :] - trajectory[:, -2, :]
    noise_scale = 0.00002
    for t in range(future_len):
        velocity_curr = velocity_curr * damping + np.random.normal(
            0, noise_scale, size=(num_agents, 2)
        )
        current_pos = current_pos + velocity_curr
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)

    # 10. Smoothed Constant Velocity with a look back of 4 frames
    velocity = (trajectory[:, -1, :] - trajectory[:, 3, :]) / 4.0
    smoothed_velocity = velocity + np.random.normal(0, 0.0001, size=(num_agents, 2))
    smoothed_traj = trajectory[:, -1:, :] + time_steps * np.expand_dims(
        smoothed_velocity, axis=1
    )
    all_trajectories.append(smoothed_traj)

    # 11. Combined Damped Momentum with Slight Acceleration
    damped = all_trajectories[1]
    acceleration = all_trajectories[3]
    combined_da = (
        0.7 * damped + 0.3 * acceleration + np.random.normal(0, 0.0001, size=damped.shape)
    )
    all_trajectories.append(combined_da)

    # 12. Circumvent with more aggressive angle change
    velocity = calculate_velocity(trajectory)
    angle = np.random.uniform(-np.pi / 30, np.pi / 30)
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated_velocity = np.dot(velocity, rotation_matrix)
    circumvent_trajectory = trajectory[:, -1:, :] + time_steps * np.expand_dims(
        rotated_velocity, axis=1
    )
    all_trajectories.append(circumvent_trajectory)

    # 13. Adaptive damping using a different velocity calculation window.
    velocity = np.median(trajectory[:, -5:, :] - trajectory[:, -6:-1, :], axis=1)
    agent_speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    damping = np.clip(1 - agent_speed * 0.009, 0.87, 0.99)
    current_pos = trajectory[:, -1, :]
    predictions = []
    velocity_curr = trajectory[:, -1, :] - trajectory[:, -2, :]
    noise_scale = 0.00004
    for t in range(future_len):
        velocity_curr = velocity_curr * damping + np.random.normal(
            0, noise_scale, size=(num_agents, 2)
        )
        current_pos = current_pos + velocity_curr
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)

    # 14. Combined: Top 3 (Damped, Adaptive, Smoothed)
    damped = all_trajectories[1]
    adaptive = all_trajectories[9]  # Use a different successful model
    smoothed = all_trajectories[2]
    combined_das = (0.45 * damped + 0.35 * adaptive + 0.2 * smoothed) + np.random.normal(
        0, 0.00015, size=damped.shape
    )
    all_trajectories.append(combined_das)

    # 15. Damped momentum with speed scaling factor.
    velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
    agent_speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    velocity = velocity * np.clip(1 + agent_speed * 0.1, 0.95, 1.05)
    damping = np.clip(0.93 - agent_speed * 0.012, 0.85, 0.98)
    current_pos = trajectory[:, -1, :]
    predictions = []
    noise_scale = 0.00005
    for t in range(future_len):
        velocity = velocity * damping + np.random.normal(
            0, noise_scale, size=(num_agents, 2)
        )
        current_pos = current_pos + velocity
        predictions.append(current_pos.copy())
    pred_trajectory = np.stack(predictions, axis=1)
    all_trajectories.append(pred_trajectory)

    # Smart Blending: Fill remaining slots by combining the best-performing models.
    # Based on offline analysis, these indices are consistent top performers.
    top_performers_indices = [0, 3, 4, 9, 11, 12]

    while len(all_trajectories) < 20:
        # Pick two different top performers to blend
        idx1, idx2 = np.random.choice(top_performers_indices, 2, replace=False)

        traj1 = all_trajectories[idx1]
        traj2 = all_trajectories[idx2]

        # Use random weights for more variety in blending
        w = np.random.uniform(0.3, 0.7)
        avg_traj = (w * traj1 + (1 - w) * traj2) + np.random.normal(
            0, 0.00005, size=traj1.shape
        )
        all_trajectories.append(avg_traj)

    return np.stack(all_trajectories[:20], axis=0)
