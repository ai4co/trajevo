import numpy as np

from pysocialforce import Simulator


def predict_trajectory(trajectory: np.ndarray, sample=True) -> np.ndarray:
    """Generate 20 diverse possible future trajectories using social force model with stochastic sampling.

    Args:
        trajectory (np.ndarray): Input trajectory of shape [num_agents, traj_length, 2], where traj_length is 8.

    Returns:
        np.ndarray: An array of 20 diverse trajectories of shape [20, num_agents, 12, 2].
    """
    num_agents = trajectory.shape[0]
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
        rotation_mat = np.array([[c, s], [-s, c]])

        # Prepare initial state for social force model
        initial_state = []
        for agent_idx in range(num_agents):
            # Get the last position and velocity
            last_pos = trajectory[agent_idx, -1]
            last_vel = trajectory[agent_idx, -1] - trajectory[agent_idx, -2]

            # Apply rotation to the velocity
            rotated_vel = np.dot(rotation_mat, last_vel)

            # Calculate goal position (assuming it's in the direction of movement)
            goal = last_pos + rotated_vel * 12  # 12 steps ahead

            # Create state vector [px, py, vx, vy, gx, gy]
            state = np.concatenate([last_pos, rotated_vel, goal])
            initial_state.append(state)

        initial_state = np.array(initial_state)

        # Create simulator
        s = Simulator(
            initial_state,
            groups=None,  # No predefined groups
            obstacles=None,  # No obstacles
        )

        # Run simulation for 12 steps
        s.step(12)

        # Get the predicted trajectories
        states, _ = s.get_states()
        predicted_trajectories = states[
            -12:, :, :2
        ]  # Get last 12 steps, all agents, x,y positions

        # Reshape to match the required output format
        sample_prediction = np.transpose(
            predicted_trajectories, (1, 0, 2)
        )  # [num_agents, 12, 2]
        all_trajectories.append(sample_prediction)

    # Stack all samples
    result = np.stack(all_trajectories, axis=0)  # [20, num_agents, 12, 2]
    return result
