import numpy as np


def predict_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Generate 20 possible future trajectories with diverse heuristics and motion patterns.

    Args:
        trajectory (np.ndarray): Input trajectory of shape [num_agents, traj_length, 2], where traj_length is 8.

    Returns:
        np.ndarray: 20 diverse trajectories of shape [20, num_agents, 12, 2].
    """
    num_agents = trajectory.shape[0]
    all_trajectories = []
    traj_length = trajectory.shape[1]

    for i in range(20):
        predictions = []
        current_pos = trajectory[:, -1, :].copy()
        velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
        acceleration = (
            velocity - (trajectory[:, -2, :] - trajectory[:, -3, :])
            if traj_length > 3
            else np.zeros_like(velocity)
        )
        jerk = (
            acceleration
            - (trajectory[:, -3, :] - 2 * trajectory[:, -4, :] + trajectory[:, -5, :])
            if traj_length > 4
            else np.zeros_like(acceleration)
        )
        snap = (
            jerk
            - (
                trajectory[:, -4, :]
                - 3 * trajectory[:, -5, :]
                + 3 * trajectory[:, -6, :]
                - trajectory[:, -7, :]
            )
            if traj_length > 6
            else np.zeros_like(jerk)
        )
        curvature = (
            np.mean(
                np.abs(
                    np.diff(
                        np.arctan2(
                            np.diff(trajectory[:, :, 1], axis=1),
                            np.diff(trajectory[:, :, 0], axis=1),
                        ),
                        axis=1,
                    )
                ),
                axis=1,
                keepdims=True,
            )
            + 1e-8
        )

        # Option 1: Linear extrapolation with adaptive noise and dynamic jerk/snap - Improved range
        if i < 3:
            noise_scale = np.random.uniform(0.01, 0.25)
            speed_change = np.abs(
                np.linalg.norm(
                    trajectory[:, -1, :] - trajectory[:, -2, :], axis=1, keepdims=True
                )
                - np.linalg.norm(
                    trajectory[:, -2, :] - trajectory[:, -3, :], axis=1, keepdims=True
                )
            )
            jerk_magnitude = np.linalg.norm(jerk, axis=1, keepdims=True)
            snap_magnitude = np.linalg.norm(snap, axis=1, keepdims=True)

            adaptive_noise = noise_scale * (
                np.linalg.norm(velocity, axis=1, keepdims=True)
                + 0.02
                + curvature
                + speed_change
                + jerk_magnitude * 0.5
                + snap_magnitude * 0.25
            )

            jerk_factor = np.random.uniform(0.002, 0.005)
            snap_factor = np.random.uniform(0.0005, 0.0015)

            for t in range(12):
                noise = np.random.randn(num_agents, 2) * adaptive_noise
                current_pos += velocity + noise
                velocity += (
                    0.01 * acceleration
                    + jerk_factor * jerk
                    + snap_factor * snap
                    + 0.0005 * np.random.randn(num_agents, 2)
                )
                acceleration += 0.001 * snap + 0.0001 * np.random.randn(num_agents, 2)
                jerk += 0.00001 * np.random.randn(num_agents, 2)
                snap += 0.000001 * np.random.randn(num_agents, 2)
                predictions.append(current_pos.copy())

        # Option 2: Averaged velocity with adaptive noise and history weighting
        elif i < 6:
            avg_velocity = np.mean(np.diff(trajectory, axis=1), axis=1)
            jerk_magnitude = np.linalg.norm(jerk, axis=1, keepdims=True)
            snap_magnitude = np.linalg.norm(snap, axis=1, keepdims=True)

            noise_scale = np.random.uniform(0.005, 0.06)
            adaptive_noise = noise_scale * (
                np.linalg.norm(avg_velocity, axis=1, keepdims=True)
                + jerk_magnitude * 0.8
                + snap_magnitude * 0.4
                + 0.01
                + curvature * 0.2
            )
            history_weight = np.random.uniform(0.05, 0.35)

            for t in range(12):
                noise = np.random.randn(num_agents, 2) * adaptive_noise
                current_pos += avg_velocity + noise
                if traj_length > 1:
                    avg_velocity = (
                        (1 - history_weight) * avg_velocity
                        + history_weight
                        * np.mean(
                            np.diff(trajectory[:, max(0, t - 7) :, :], axis=1), axis=1
                        )
                        if t < 7
                        else avg_velocity
                        + 0.003 * jerk
                        + 0.001 * snap
                        + 0.0005 * np.random.randn(num_agents, 2)
                    )
                else:
                    avg_velocity = (
                        avg_velocity
                        + 0.003 * jerk
                        + 0.001 * snap
                        + 0.0005 * np.random.randn(num_agents, 2)
                    )

                jerk += 0.0001 * np.random.randn(num_agents, 2)

                predictions.append(current_pos.copy())

        # Option 3: Random direction changes (refined and widened)
        elif i < 12:
            angle_scale = np.random.uniform(np.pi / 60, np.pi / 15)
            smooth_factor = np.random.uniform(0.70, 0.99)
            velocity_decay = np.random.uniform(0.95, 1.0)
            angular_noise = np.random.uniform(
                -0.005, 0.005, size=(num_agents, 1)
            )  # small angular noise

            for t in range(12):
                if t % 3 == 0:
                    angle = (
                        np.random.uniform(
                            -angle_scale * (1 + 0.2 * curvature),
                            angle_scale * (1 + 0.2 * curvature),
                            size=(num_agents, 1),
                        )
                        + angular_noise
                    )
                    rotation_matrix = np.concatenate(
                        [np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)],
                        axis=1,
                    ).reshape(num_agents, 2, 2)
                    velocity = np.matmul(
                        velocity[:, np.newaxis, :], rotation_matrix
                    ).squeeze()
                    velocity = smooth_factor * velocity + (1 - smooth_factor) * (
                        trajectory[:, -1, :] - trajectory[:, -2, :]
                    )

                velocity *= velocity_decay
                current_pos += velocity
                predictions.append(current_pos.copy())

        # Option 4: Gradual Stop (Adaptive, noise-enhanced)
        elif i < 17:
            decay_rate = np.random.uniform(0.90, 0.998)
            min_speed = np.random.uniform(0.0005, 0.025)
            noise_scale = np.random.uniform(0.001, 0.01)
            initial_speed = np.linalg.norm(
                trajectory[:, -1, :] - trajectory[:, -2, :], axis=1, keepdims=True
            )
            dynamic_decay = 1 - (1 - decay_rate) * (
                1 + initial_speed * np.random.uniform(0.2, 0.8)
            )  # Initial speed affects how fast decay changes

            for t in range(12):
                velocity *= dynamic_decay
                speed = np.linalg.norm(velocity, axis=1, keepdims=True)
                velocity = np.where(speed < min_speed, 0, velocity)
                noise = np.random.randn(num_agents, 2) * noise_scale
                current_pos += velocity + noise
                predictions.append(current_pos.copy())

        # Option 5: Jerky Motion
        elif i < 19:
            jerk_strength = np.random.uniform(-0.002, 0.002, size=(num_agents, 2))
            velocity_limit = np.random.uniform(0.5, 2.0)
            noise_scale = np.random.uniform(0.001, 0.01)

            for t in range(12):
                velocity += jerk_strength
                speed = np.linalg.norm(velocity, axis=1, keepdims=True)
                velocity = np.where(
                    speed > velocity_limit, velocity / speed * velocity_limit, velocity
                )  # Limit velocity magnitude
                noise = np.random.randn(num_agents, 2) * noise_scale
                current_pos += velocity + noise
                jerk_strength += np.random.uniform(
                    -0.0002, 0.0002, size=(num_agents, 2)
                )  # Jitter in jerk
                predictions.append(current_pos.copy())

        # Option 6: Stagnant trajectory with tiny perturbation (biased, but less biased)
        else:
            avg_velocity = np.mean(np.diff(trajectory, axis=1), axis=1)
            # Use recent velocity trend for bias
            if traj_length >= 3:
                recent_velocities = np.diff(trajectory[:, -3:, :], axis=1)
                trend_velocity = np.mean(recent_velocities, axis=1)
                bias = trend_velocity * 0.01  # Use trend velocity as bias
            else:
                bias = np.mean(avg_velocity, axis=0) * 0.005

            # Add momentum with decay
            momentum = bias.copy()
            decay_rate = 0.95

            for t in range(12):
                noise = np.random.normal(0, 0.0005, size=(num_agents, 2))
                current_pos += momentum + noise
                momentum *= decay_rate  # Momentum decays over time
                predictions.append(current_pos.copy())

        pred_trajectory = np.stack(predictions, axis=1)
        all_trajectories.append(pred_trajectory)

    all_trajectories = np.stack(all_trajectories, axis=0)
    return all_trajectories
