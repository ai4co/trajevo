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

    # Calculate acceleration for better predictions
    def calculate_acceleration(traj):
        if traj.shape[1] > 2:
            v1 = traj[:, -1, :] - traj[:, -2, :]
            v2 = traj[:, -2, :] - traj[:, -3, :]
            return v1 - v2
        else:
            return np.zeros((num_agents, 2))

    # 1. Enhanced Constant Velocity with Adaptive Gaussian Noise and Mode Switching (Dominant Strategy - 14 trajectories)
    for i in range(14):  # Increased Allocation
        velocity = calculate_velocity(trajectory)
        acceleration = calculate_acceleration(trajectory)
        current_pos = trajectory[:, -1, :].copy()
        predictions = []

        # Adaptive noise based on recent velocity magnitude and trajectory context
        velocity_magnitude = np.linalg.norm(velocity, axis=1, keepdims=True)
        mean_velocity_magnitude = np.mean(velocity_magnitude)
        if mean_velocity_magnitude < min_speed:
            mean_velocity_magnitude = min_speed

        # Context-aware noise scaling - reduced base noise
        noise_level = (
            0.003  # Reduced base noise
            + 0.005 * i * (velocity_magnitude / (mean_velocity_magnitude + 1e-8))
            + 0.002 * (8 - trajectory.shape[1]) / 8
        )

        # Enhanced Mode Switching with smoother variations
        if i < 8:  # First 8 trajectories: smooth velocity modifications
            velocity_modifier = 0.95 + 0.1 * np.sin(i * np.pi / 4)  # 0.85 to 1.05
            angle_modifier = 0.05 * np.sin(i * np.pi / 4)  # Smaller angle perturbation
        else:  # Last 6 trajectories: incorporate acceleration
            velocity_modifier = 1.0
            angle_modifier = 0.02 * (i - 8)
            # Add small acceleration component
            velocity += acceleration * 0.05 * (i - 8) / 6

        # Rotate velocity vector slightly
        angle = np.arctan2(velocity[:, 1], velocity[:, 0])
        angle += angle_modifier
        velocity[:, 0] = np.cos(angle) * np.linalg.norm(velocity, axis=1)
        velocity[:, 1] = np.sin(angle) * np.linalg.norm(velocity, axis=1)

        velocity = velocity * velocity_modifier

        # Add slight decay for long-term stability
        decay_rate = 0.998 if i < 10 else 0.995

        for step in range(prediction_length):
            # Adaptive noise that slightly increases with time
            time_factor = 1.0 + step * 0.02
            noise = np.random.normal(0, noise_level * time_factor, size=current_pos.shape)

            # Apply decay
            velocity *= decay_rate

            new_pos = current_pos + velocity + noise

            # Clip Speed
            delta = new_pos - current_pos
            speed = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, max_speed / (speed + 1e-8))
            new_pos = current_pos + delta * scale

            predictions.append(new_pos.copy())
            current_pos = new_pos.copy()

        all_trajectories.append(np.stack(predictions, axis=1))

    # 2. Momentum-based prediction (Similar to ETH Option 6 - 3 trajectories)
    for i in range(3):
        current_pos = trajectory[:, -1, :].copy()
        predictions = []

        # Use recent velocity trend
        if trajectory.shape[1] >= 3:
            recent_velocities = np.diff(trajectory[:, -3:, :], axis=1)
            trend_velocity = np.mean(recent_velocities, axis=1)
            bias = trend_velocity * (0.008 + 0.002 * i)  # Variable bias
        else:
            bias = calculate_velocity(trajectory) * 0.005

        # Add momentum with decay
        momentum = bias.copy()
        decay_rate = 0.95 - 0.01 * i  # Variable decay: 0.95, 0.94, 0.93
        noise_scale = 0.0003 + 0.0001 * i  # Variable noise

        for step in range(prediction_length):
            noise = np.random.normal(0, noise_scale, size=current_pos.shape)
            current_pos += momentum + noise
            momentum *= decay_rate  # Momentum decays over time
            predictions.append(current_pos.copy())

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
        noise_level = 0.002 + 0.003 * i  # Lower noise for smoother trajectories

        # Add momentum with stronger history weight
        if trajectory.shape[1] > 3:
            prev_velocity = np.mean(
                np.diff(trajectory[:, -4:, :], axis=1), axis=1
            )  # Average velocity of the last 3 steps
            velocity = (
                0.8 * velocity + 0.2 * prev_velocity
            )  # Stronger current velocity weight

        # Agent-specific adjustments
        for agent_id in range(num_agents):
            if np.linalg.norm(velocity[agent_id, :]) < 0.05:  # Nearly stationary
                velocity[agent_id, :] *= 0.5  # Further reduce velocity
            elif np.random.rand() < 0.1:  # 10% chance of slight deviation
                velocity[agent_id, 0] += np.random.uniform(-0.1, 0.1)
                velocity[agent_id, 1] += np.random.uniform(-0.1, 0.1)

        for step in range(prediction_length):
            noise = np.random.normal(0, noise_level, size=current_pos.shape)
            new_pos = current_pos + velocity + noise

            # Clip Speed
            delta = new_pos - current_pos
            speed = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, max_speed / (speed + 1e-8))
            new_pos = current_pos + delta * scale

            # Slight velocity decay
            velocity *= 0.997

            predictions.append(new_pos.copy())
            current_pos = new_pos.copy()
        all_trajectories.append(np.stack(predictions, axis=1))

    # 4. Adaptive Random Walk (Keep for diversity - 1 trajectory)
    for i in range(1):
        current_pos = trajectory[:, -1, :].copy()
        predictions = []

        # Get initial velocity for biased random walk
        initial_velocity = calculate_velocity(trajectory)

        # Adapt noise based on trajectory length and velocity
        noise_level = 0.003 * (8 / trajectory.shape[1])  # Reduced noise
        velocity_magnitude = np.linalg.norm(initial_velocity, axis=1, keepdims=True)
        noise_level *= 1 - np.clip(velocity_magnitude / max_speed, 0, 0.8)

        for step in range(prediction_length):
            # Biased random walk - slightly favor the initial direction
            bias = initial_velocity * 0.1 * (1 - step / prediction_length)
            noise = np.random.normal(0, noise_level, size=current_pos.shape)
            new_pos = current_pos + bias + noise

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
