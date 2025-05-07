import numpy as np


def predict_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Generate 20 possible future trajectories using diverse heuristics, with enhanced adaptive noise, mode switching, and agent-specific behaviors.

    Args:
        trajectory (np.ndarray): Input trajectory of shape [num_agents, traj_length, 2], where traj_length is 8.

    Returns:
        np.ndarray: 20 diverse trajectories of shape [20, num_agents, 12, 2].
    """

    num_agents = trajectory.shape[0]
    all_trajectories = []
    max_speed = 3.0  # Limit max speed to have plausible trajectories
    min_speed = 0.1  # Minimum speed to avoid divisions by zero
    prediction_length = 12

    # Helper function to calculate velocity with robust handling of short trajectories
    def calculate_velocity(traj):
        if traj.shape[1] > 1:
            return traj[:, -1, :] - traj[:, -2, :]
        else:
            return np.zeros((num_agents, 2))

    # 1. Enhanced Constant Velocity with Adaptive Gaussian Noise and Mode Switching (Dominant Strategy - 14 trajectories)
    for i in range(14):  # Increased Allocation
        velocity = calculate_velocity(trajectory)
        current_pos = trajectory[:, -1, :].copy()
        predictions = []

        # Adaptive noise based on recent velocity magnitude and trajectory context
        velocity_magnitude = np.linalg.norm(velocity, axis=1, keepdims=True)
        mean_velocity_magnitude = np.mean(velocity_magnitude)
        if mean_velocity_magnitude < min_speed:
            mean_velocity_magnitude = min_speed

        # Context-aware noise scaling
        noise_level = (
            0.005
            + 0.01 * i * (velocity_magnitude / (mean_velocity_magnitude + 1e-8))
            + 0.003 * (8 - trajectory.shape[1]) / 8
        )

        # Enhanced Mode Switching:  Sinusoidal velocity and direction modifier
        velocity_modifier = 0.8 + 0.4 * np.sin(
            i * np.pi / 6
        )  # Variation between 0.8 and 1.2
        angle_modifier = 0.1 * np.sin(i * np.pi / 3)  # Small angle perturbation

        # Rotate velocity vector slightly
        angle = np.arctan2(velocity[:, 1], velocity[:, 0])
        angle += angle_modifier
        velocity[:, 0] = np.cos(angle) * np.linalg.norm(velocity, axis=1)
        velocity[:, 1] = np.sin(angle) * np.linalg.norm(velocity, axis=1)

        velocity = velocity * velocity_modifier

        for step in range(prediction_length):
            noise = np.random.normal(0, noise_level, size=current_pos.shape)
            new_pos = current_pos + velocity + noise

            # Clip Speed
            delta = new_pos - current_pos
            speed = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, max_speed / (speed + 1e-8))
            new_pos = current_pos + delta * scale

            predictions.append(new_pos.copy())
            current_pos = new_pos.copy()

        all_trajectories.append(np.stack(predictions, axis=1))

    # 2. Constant Velocity with Adaptive Laplacian Noise (For Diversity - 2 trajectories)
    for i in range(2):
        velocity = calculate_velocity(trajectory)
        current_pos = trajectory[:, -1, :].copy()
        predictions = []
        velocity_magnitude = np.linalg.norm(velocity, axis=1, keepdims=True)
        mean_velocity_magnitude = np.mean(velocity_magnitude)
        if mean_velocity_magnitude < min_speed:
            mean_velocity_magnitude = min_speed
        noise_level = 0.015 + 0.02 * i * (
            velocity_magnitude / (mean_velocity_magnitude + 1e-8)
        )

        # Agent-specific adjustments:
        for agent_id in range(num_agents):
            if (
                np.linalg.norm(trajectory[agent_id, -1, :]) < 0.3
            ):  # More sensitive Stationary Agent
                velocity[agent_id, :] = 0.0  # make agent stationary, more certain
            elif np.random.rand() < 0.15:  # 15% chance of slight deviation
                velocity[agent_id, 0] += np.random.uniform(-0.2, 0.2)
                velocity[agent_id, 1] += np.random.uniform(-0.2, 0.2)

        for step in range(prediction_length):
            noise = np.random.laplace(0, noise_level, size=current_pos.shape)
            new_pos = current_pos + velocity + noise

            # Clip Speed
            delta = new_pos - current_pos
            speed = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, max_speed / (speed + 1e-8))
            new_pos = current_pos + delta * scale

            predictions.append(new_pos.copy())
            current_pos = new_pos.copy()
        all_trajectories.append(np.stack(predictions, axis=1))

    # 3. Smoothed Velocity with Lower Gaussian Noise (Stable Trajectories - 2 trajectories)
    for i in range(2):
        velocity = (
            np.mean(np.diff(trajectory, axis=1), axis=1)
            if trajectory.shape[1] > 1
            else np.zeros((num_agents, 2))
        )  # Smoothed velocity
        current_pos = trajectory[:, -1, :].copy()
        predictions = []
        noise_level = 0.003 + 0.007 * i  # Lower noise for smoother trajectories

        # Add momentum:
        if trajectory.shape[1] > 3:
            prev_velocity = np.mean(
                np.diff(trajectory[:, -4:, :], axis=1), axis=1
            )  # Average velocity of the last 3 steps
            velocity = 0.7 * velocity + 0.3 * prev_velocity  # Use weighted average

        for step in range(prediction_length):
            noise = np.random.normal(0, noise_level, size=current_pos.shape)
            new_pos = current_pos + velocity + noise

            # Clip Speed
            delta = new_pos - current_pos
            speed = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, max_speed / (speed + 1e-8))
            new_pos = current_pos + delta * scale

            predictions.append(new_pos.copy())
            current_pos = new_pos.copy()
        all_trajectories.append(np.stack(predictions, axis=1))

    # 4. Random Walk (Keep for diversity, even lower allocation - 2 trajectories)
    for i in range(2):
        current_pos = trajectory[:, -1, :].copy()
        predictions = []

        # Adapt noise based on trajectory length and velocity. Reduced noise levels for more plausible random walks
        noise_level = (0.005 + 0.003 * i) * (
            8 / trajectory.shape[1]
        )  # Further reduced noise
        velocity_magnitude = np.linalg.norm(
            calculate_velocity(trajectory), axis=1, keepdims=True
        )
        noise_level *= 1 - np.clip(
            velocity_magnitude / max_speed, 0, 1
        )  # Reduce Noise if agents are already moving fast

        for step in range(prediction_length):
            noise = np.random.normal(0, noise_level, size=current_pos.shape)
            new_pos = current_pos + noise

            # Clip Speed
            delta = new_pos - current_pos
            speed = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, max_speed / (speed + 1e-8))
            new_pos = current_pos + delta * scale

            predictions.append(new_pos.copy())
            current_pos = new_pos.copy()
        all_trajectories.append(np.stack(predictions, axis=1))

    all_trajectories = np.stack(
        all_trajectories[:20], axis=0
    )  # Ensure exactly 20 trajectories.
    return all_trajectories
