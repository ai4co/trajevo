import numpy as np


def predict_trajectory(trajectory: np.ndarray, sample=True) -> np.ndarray:
    """
    Generate 20 diverse possible future trajectories using CSCRCTR model.

    Args:
        trajectory (np.ndarray): [num_agents, traj_length, 2], traj_length = 8
        sample (bool): Whether to add angular noise to predictions.

    Returns:
        np.ndarray: [20, num_agents, 12, 2] predicted future trajectories
    """

    def _init_state_from_traj(traj: np.ndarray) -> np.ndarray:
        """
        Initialize the CSCRCTR state from historical trajectory.

        Args:
            traj (np.ndarray): [8,2] historical trajectory.

        Returns:
            np.ndarray: [x, y, phi_c, phi_p, d_c, d_p] state.
        """
        A, B, C = traj[-3], traj[-2], traj[-1]
        gamma = np.linalg.norm(B - A)
        beta = np.linalg.norm(C - A)
        alpha = np.linalg.norm(C - B)
        s = (alpha + beta + gamma) / 2
        area = np.sqrt(max(s * (s - alpha) * (s - beta) * (s - gamma), 0))
        if area < 1e-6 or abs(alpha + gamma - beta) < 1e-6:
            r = 1e6  # approximately a straight line
            theta1 = theta2 = 0
        else:
            r = alpha * beta * gamma / (4 * area)
            theta1 = np.arcsin(np.clip(gamma / (2 * r), -1, 1))
            theta2 = np.arcsin(np.clip(alpha / (2 * r), -1, 1))
        d_c = 2 * r * theta1 if area > 1e-6 else alpha
        d_p = 2 * r * theta2 if area > 1e-6 else gamma
        delta = C - B
        eta_BC = np.arctan2(delta[1], delta[0])
        cross = (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])
        if cross < 0:  # clockwise
            phi_c = eta_BC - theta1
            phi_p = eta_BC + theta1
        else:  # counter-clockwise or straight
            phi_c = eta_BC + theta1
            phi_p = eta_BC - theta1
        x, y = C
        return np.array([x, y, phi_c, phi_p, d_c, d_p])

    def _cscrctr_step(state: np.ndarray, dt=1.0, u=0.0, v=0.0) -> np.ndarray:
        """
        Single step propagation of CSCRCTR model.

        Args:
            state (np.ndarray): [x, y, phi_c, phi_p, d_c, d_p].
            dt (float): time step, default 1.0.
            u (float): heading noise.
            v (float): distance noise.

        Returns:
            np.ndarray: updated state.
        """
        x, y, phi_c, phi_p, d_c, d_p = state
        phi_c_new = 2 * phi_c - phi_p + u
        phi_p_new = phi_c
        d_c_new = 2 * d_c - d_p + v
        d_p_new = d_c
        delta_phi = phi_c_new - phi_p_new
        if abs(delta_phi) < 1e-6:
            dx = d_c_new * np.cos(phi_c_new)
            dy = d_c_new * np.sin(phi_c_new)
        else:
            dx = d_c_new * (np.sin(phi_c_new) - np.sin(phi_p_new)) / delta_phi
            dy = -d_c_new * (np.cos(phi_c_new) - np.cos(phi_p_new)) / delta_phi
        x_new = x + dx
        y_new = y + dy
        return np.array([x_new, y_new, phi_c_new, phi_p_new, d_c_new, d_p_new])

    num_agents = trajectory.shape[0]
    future_steps = 12
    num_samples = 20
    preds = np.zeros((num_samples, num_agents, future_steps, 2), dtype=np.float32)
    dt = 1.0
    sigma_phi = 0.02  # angular noise
    sigma_d = 0.02  # distance noise

    for agent in range(num_agents):
        init_state = _init_state_from_traj(trajectory[agent])
        for s in range(num_samples):
            state = init_state.copy()
            if sample:
                state[2] += np.random.normal(0, sigma_phi)
                state[3] += np.random.normal(0, sigma_phi)
                state[4] += np.random.normal(0, sigma_d)
                state[5] += np.random.normal(0, sigma_d)
            for t in range(future_steps):
                u = np.random.normal(0, sigma_phi) if sample else 0.0
                v = np.random.normal(0, sigma_d) if sample else 0.0
                state = _cscrctr_step(state, dt, u, v)
                preds[s, agent, t] = state[:2]

    return preds
