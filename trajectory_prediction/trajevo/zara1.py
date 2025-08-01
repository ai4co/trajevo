import numpy as np


def predict_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Generate 20 possible future trajectories with enhanced diversification and adaptive strategies.

    Args:
        trajectory (np.ndarray): [num_agents, traj_length, 2] where traj_length is 8.

    Returns:
        np.ndarray: 20 diverse trajectories [20, num_agents, 12, 2].
    """
    num_agents = trajectory.shape[0]
    all_trajectories = []
    history_len = trajectory.shape[1]

    # Calculate average speed from history for adaptive parameters
    # Avoid division by zero if history_len is 1
    if history_len > 1:
        historical_velocities = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        avg_speed_history = np.mean(np.linalg.norm(historical_velocities, axis=2))
    else:
        avg_speed_history = 0.0

    for i in range(20):
        current_pos = trajectory[:, -1, :]

        # Option 1: Dominant strategy - Average velocity with adaptive noise, rotation, and parameter variation
        if i < 14:  # Increased to 14, best performing strategy
            velocity = np.zeros_like(current_pos)
            weights_sum = 0.0
            # Adaptive decay rate based on historical speed variability
            # Higher variability -> faster decay (less reliance on old history)
            # Lower variability -> slower decay (more reliance on old history)
            speed_std = np.std(avg_speed_history) if history_len > 1 else 0.0
            decay_rate = np.clip(0.1 + speed_std * 0.2, 0.1, 0.4)  # Adjusted range

            # Consider more history for velocity calculation if available
            # Increased max history from 5 to 7 for smoother average
            for k in range(min(history_len - 1, 7)):
                weight = np.exp(-decay_rate * k)
                velocity += weight * (trajectory[:, -1 - k, :] - trajectory[:, -2 - k, :])
                weights_sum += weight
            velocity /= weights_sum + 1e-8

            # avg_speed used for noise and rotation scales is for the current prediction step,
            # using average of past velocities is more robust.
            current_avg_speed = np.mean(np.linalg.norm(velocity, axis=1))

            noise_scale = (
                0.012 + current_avg_speed * 0.007
            )  # Slightly reduced avg_speed influence on noise
            noise = np.random.normal(0, noise_scale, size=(num_agents, 12, 2))

            angle = np.random.uniform(-0.05, 0.05)
            rotation_matrix = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            velocity = velocity @ rotation_matrix

            momentum = 0.0
            jerk_factor = 0.0
            damping = 0.0

            # Parameter Variation
            if i % 6 == 0:
                noise_scale *= np.random.uniform(
                    0.9, 1.1
                )  # Fine-tuned noise scale variation
                noise = np.random.normal(0, noise_scale, size=(num_agents, 12, 2))
            elif i % 6 == 1:
                angle_scale = 0.06 + current_avg_speed * 0.02
                angle = np.random.uniform(
                    -angle_scale * np.random.uniform(0.8, 1.2),
                    angle_scale * np.random.uniform(0.8, 1.2),
                )
                rotation_matrix = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                velocity = velocity @ rotation_matrix
            elif i % 6 == 2:
                momentum = np.random.uniform(
                    0.07, 0.15
                )  # Slightly adjusted momentum range
                velocity = momentum * velocity + (1 - momentum) * (
                    trajectory[:, -1, :] - trajectory[:, -2, :]
                )
            elif i % 6 == 3:  # Add jerk
                jerk_factor = np.random.uniform(0.003, 0.007)  # Adjusted jerk factor
                if history_len > 2:
                    jerk = (
                        trajectory[:, -1, :]
                        - 2 * trajectory[:, -2, :]
                        + trajectory[:, -3, :]
                    )
                else:
                    jerk = np.zeros_like(velocity)
                velocity += jerk_factor * jerk
            elif i % 6 == 4:  # Damping
                damping = np.random.uniform(0.007, 0.02)  # Adjusted damping range
                velocity = velocity * (1 - damping)
            else:  # Adaptive Noise Scale
                noise_scale = 0.01 + current_avg_speed * np.random.uniform(
                    0.007, 0.015
                )  # Adjusted adaptive noise
                noise = np.random.normal(0, noise_scale, size=(num_agents, 12, 2))

            predictions = []
            for t in range(1, 13):
                # Adjusted noise decay exponent for potentially better ADE/FDE
                current_pos = current_pos + velocity + noise[:, t - 1, :] / (t**0.5)
                predictions.append(current_pos.copy())
            pred_trajectory = np.stack(predictions, axis=1)

        # Option 2: Velocity rotation with adaptive angle
        elif i < 17:  # Increased to 17.
            velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
            current_avg_speed = np.mean(
                np.linalg.norm(velocity, axis=1)
            )  # Using current velocity for avg_speed
            angle_scale = (
                0.12 + current_avg_speed * 0.045
            )  # adaptive angle, slightly reduced influence

            angle = np.random.uniform(-angle_scale, angle_scale)  # adaptive range
            rotation_matrix = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            velocity = velocity @ rotation_matrix

            noise_scale = (
                0.006 + current_avg_speed * 0.0035
            )  # Slightly reduced noise scale
            noise = np.random.normal(0, noise_scale, size=(num_agents, 12, 2))

            predictions = []
            for t in range(1, 13):
                # Adjusted noise decay exponent
                current_pos = current_pos + velocity + noise[:, t - 1, :] / (t**0.6)
                predictions.append(current_pos.copy())
            pred_trajectory = np.stack(predictions, axis=1)

        # Option 3: Memory-based approach (repeating last velocity) + Enhanced Collision Avoidance
        elif i < 19:  # Increased to 19
            velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
            # Enhanced smoothing with more velocity history and slightly adjusted weights
            if history_len > 3:
                velocity = (
                    0.5 * velocity  # Reduced weight for immediate last velocity
                    + 0.35
                    * (
                        trajectory[:, -2, :] - trajectory[:, -3, :]
                    )  # Increased weight for older velocity
                    + 0.15 * (trajectory[:, -3, :] - trajectory[:, -4, :])
                )
            elif history_len > 2:
                velocity = 0.6 * velocity + 0.4 * (  # Adjusted weights
                    trajectory[:, -2, :] - trajectory[:, -3, :]
                )
            # else: velocity remains as is for history_len <= 2

            current_avg_speed = np.mean(np.linalg.norm(velocity, axis=1))

            # Adaptive Laplacian noise
            noise_scale = (
                0.004 + current_avg_speed * 0.001
            )  # Slightly reduced noise scale
            noise = np.random.laplace(0, noise_scale, size=(num_agents, 2))
            velocity = velocity + noise

            # Enhanced collision avoidance
            repulsion_strength = (
                0.0013  # Slightly increased repulsion strength for clearer separation
            )
            predictions = []
            temp_pos = current_pos.copy()

            for t in range(1, 13):
                net_repulsions = np.zeros_like(temp_pos)
                for agent_idx in range(num_agents):
                    for other_idx in range(num_agents):
                        if agent_idx != other_idx:
                            direction = temp_pos[agent_idx] - temp_pos[other_idx]
                            distance = np.linalg.norm(direction)
                            # Adjusted interaction threshold and added a soft clamping to repulsion
                            if distance < 1.1:
                                repulsion = (
                                    (direction / (distance + 1e-6))
                                    * repulsion_strength
                                    * np.exp(
                                        -distance * 2
                                    )  # Stronger exponential decay for closer proximity
                                )
                                # Apply a small dampening to prevent overshooting due to strong repulsion
                                repulsion = np.clip(repulsion, -0.05, 0.05)
                                net_repulsions[agent_idx] += repulsion

                # Adjusted balance between current velocity and repulsion for smoother collision avoidance
                velocity = (
                    0.85 * velocity + 0.15 * net_repulsions
                )  # More influence from repulsion
                temp_pos = temp_pos + velocity
                predictions.append(temp_pos.copy())

            pred_trajectory = np.stack(predictions, axis=1)

        # Option 4: Linear prediction with adaptive damping and larger noise.
        else:
            velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
            damping = np.random.uniform(
                0.015, 0.035
            )  # Slightly adjusted damping factor range

            noise_scale = 0.025  # Slightly reduced base noise scale
            noise = np.random.normal(0, noise_scale, size=(num_agents, 12, 2))

            predictions = []
            for t in range(1, 13):
                velocity = velocity * (1 - damping) + noise[:, t - 1, :] / (
                    t**0.5  # Adjusted noise decay exponent
                )
                current_pos = current_pos + velocity
                predictions.append(current_pos.copy())
            pred_trajectory = np.stack(predictions, axis=1)

        all_trajectories.append(pred_trajectory)

    all_trajectories = np.stack(all_trajectories, axis=0)
    return all_trajectories
