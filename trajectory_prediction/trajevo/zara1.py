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

    for i in range(20):
        current_pos = trajectory[:, -1, :]

        # Option 1: Dominant strategy - Average velocity with adaptive noise, rotation, and parameter variation
        if i < 14:  # Increased to 14, best performing strategy
            velocity = np.zeros_like(current_pos)
            weights_sum = 0.0
            decay_rate = np.random.uniform(0.1, 0.3)  # Adaptive decay rate
            for k in range(min(history_len - 1, 5)):
                weight = np.exp(-decay_rate * k)
                velocity += weight * (trajectory[:, -1 - k, :] - trajectory[:, -2 - k, :])
                weights_sum += weight
            velocity /= weights_sum + 1e-8

            avg_speed = np.mean(np.linalg.norm(velocity, axis=1))
            noise_scale = 0.012 + avg_speed * 0.008
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
                angle_scale = 0.06 + avg_speed * 0.02
                angle = np.random.uniform(
                    -angle_scale * np.random.uniform(0.8, 1.2),
                    angle_scale * np.random.uniform(0.8, 1.2),
                )
                rotation_matrix = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                velocity = velocity @ rotation_matrix
            elif i % 6 == 2:
                momentum = np.random.uniform(0.06, 0.14)  # Vary momentum
                velocity = momentum * velocity + (1 - momentum) * (
                    trajectory[:, -1, :] - trajectory[:, -2, :]
                )
            elif i % 6 == 3:  # Add jerk
                jerk_factor = np.random.uniform(0.0025, 0.0065)
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
                damping = np.random.uniform(0.006, 0.019)
                velocity = velocity * (1 - damping)
            else:  # Adaptive Noise Scale
                noise_scale = 0.01 + avg_speed * np.random.uniform(0.006, 0.014)
                noise = np.random.normal(0, noise_scale, size=(num_agents, 12, 2))

            predictions = []
            for t in range(1, 13):
                current_pos = current_pos + velocity + noise[:, t - 1, :] / (t**0.4)
                predictions.append(current_pos.copy())
            pred_trajectory = np.stack(predictions, axis=1)

        # Option 2: Velocity rotation with adaptive angle
        elif i < 17:  # Increased to 17.
            velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
            avg_speed = np.mean(np.linalg.norm(velocity, axis=1))
            angle_scale = 0.13 + avg_speed * 0.05  # adaptive angle

            angle = np.random.uniform(-angle_scale, angle_scale)  # adaptive range
            rotation_matrix = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            velocity = velocity @ rotation_matrix

            noise_scale = 0.007 + avg_speed * 0.004
            noise = np.random.normal(0, noise_scale, size=(num_agents, 12, 2))

            predictions = []
            for t in range(1, 13):
                current_pos = current_pos + velocity + noise[:, t - 1, :] / (t**0.5)
                predictions.append(current_pos.copy())
            pred_trajectory = np.stack(predictions, axis=1)

        # Option 3: Memory-based approach (repeating last velocity) + Enhanced Collision Avoidance
        elif i < 19:  # Increased to 19
            velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
            # Enhanced smoothing with more velocity history
            if history_len > 3:
                velocity = (
                    0.55 * velocity
                    + 0.3 * (trajectory[:, -2, :] - trajectory[:, -3, :])
                    + 0.15 * (trajectory[:, -3, :] - trajectory[:, -4, :])
                )
            elif history_len > 2:
                velocity = 0.65 * velocity + 0.35 * (
                    trajectory[:, -2, :] - trajectory[:, -3, :]
                )
            else:
                velocity = velocity  # do nothing

            avg_speed = np.mean(np.linalg.norm(velocity, axis=1))

            # Adaptive Laplacian noise
            noise_scale = 0.005 + avg_speed * 0.0015
            noise = np.random.laplace(0, noise_scale, size=(num_agents, 2))
            velocity = velocity + noise

            # Enhanced collision avoidance
            repulsion_strength = 0.0011  # Adjusted repulsion strength
            predictions = []
            temp_pos = current_pos.copy()

            # Store predicted positions for efficient collision calculation at each timestep
            future_positions = [temp_pos.copy()]  # Start with current position
            for t in range(1, 13):
                net_repulsions = np.zeros_like(temp_pos)
                for agent_idx in range(num_agents):
                    for other_idx in range(num_agents):
                        if agent_idx != other_idx:
                            direction = temp_pos[agent_idx] - temp_pos[other_idx]
                            distance = np.linalg.norm(direction)
                            if distance < 1.05:  # Adjusted interaction threshold
                                repulsion = (
                                    (direction / (distance + 1e-6))
                                    * repulsion_strength
                                    * np.exp(-distance)
                                )  # distance-based decay
                                net_repulsions[agent_idx] += repulsion

                velocity = (
                    0.9 * velocity + 0.1 * net_repulsions
                )  # Damping the change in velocity
                temp_pos = temp_pos + velocity
                future_positions.append(
                    temp_pos.copy()
                )  # Store for future repulsion calculations
                predictions.append(temp_pos.copy())

            pred_trajectory = np.stack(predictions, axis=1)

        # Option 4: Linear prediction with adaptive damping and larger noise.
        else:
            velocity = trajectory[:, -1, :] - trajectory[:, -2, :]
            damping = np.random.uniform(0.017, 0.038)  # damping factor

            noise_scale = 0.028
            noise = np.random.normal(0, noise_scale, size=(num_agents, 12, 2))

            predictions = []
            for t in range(1, 13):
                velocity = velocity * (1 - damping) + noise[:, t - 1, :] / (
                    t**0.4
                )  # damping
                current_pos = current_pos + velocity
                predictions.append(current_pos.copy())
            pred_trajectory = np.stack(predictions, axis=1)

        all_trajectories.append(pred_trajectory)

    all_trajectories = np.stack(all_trajectories, axis=0)
    return all_trajectories
