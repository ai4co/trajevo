# under construction
import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch

from trajectory_prediction.utils import compute_batch_ade, compute_batch_fde

try:
    from trajectory_prediction.gpt import predict_trajectory_v2 as predict_trajectory
except:
    from trajectory_prediction.gpt import predict_trajectory

# Define constants
COLORS = [
    np.array([147, 0, 210, 255]) / 255.0,
    np.array([220, 19, 59, 255]) / 255.0,
    np.array([255, 139, 0, 255]) / 255.0,
    np.array([123, 251, 0, 255]) / 255.0,
]


def convert_coordinate(x, y, H, data_type):

    if data_type == "sdd":
        u, v = x, y
        u *= 100
        v *= 100
    else:
        ones = np.ones_like(x)
        points = np.stack([x, y, ones], axis=-1)  # shape: (N, 3)

        img_coords = points @ H.T  # shape: (N, 3)

        u = img_coords[..., 0] / img_coords[..., 2]
        v = img_coords[..., 1] / img_coords[..., 2]

        if data_type in ["eth", "hotel"]:
            u, v = v, u

    return u, v


def load_data(file_path):
    """Load and process trajectory data from file"""
    data = np.loadtxt(file_path)

    # Extract unique frame IDs and agent IDs
    frame_ids = np.unique(data[:, 0])
    agent_ids = np.unique(data[:, 1])

    print(f"Data shape: {data.shape}")
    print(f"Number of frames: {len(frame_ids)}")
    print(f"Number of agents: {len(agent_ids)}")

    # Organize data into a dictionary keyed by agent
    trajectories = {}
    for entry in data:
        frame_id, agent_id, x, y = entry
        if agent_id not in trajectories:
            trajectories[agent_id] = {}
        trajectories[agent_id][frame_id] = (x, y)

    return trajectories, frame_ids, agent_ids


def find_multiple_valid_scenarios(
    trajectories, frame_ids, num_scenarios=3, history_len=8, future_len=12
):
    """Find frames and agents with sufficient historical and future trajectory data"""
    valid_scenarios = []

    for frame_idx in range(len(frame_ids) - history_len - future_len + 1):
        current_frame = frame_ids[frame_idx + history_len - 1]

        # Ensure we have enough historical and future frames
        history_frames = frame_ids[frame_idx : frame_idx + history_len]
        future_frames = frame_ids[
            frame_idx + history_len : frame_idx + history_len + future_len
        ]

        # Check which agents have data in all these frames
        valid_agents = []
        for agent_id in trajectories:
            agent_frames = set(trajectories[agent_id].keys())
            if all(frame in agent_frames for frame in history_frames) and all(
                frame in agent_frames for frame in future_frames
            ):
                valid_agents.append(agent_id)

        if len(valid_agents) >= 1:  # At least one agent has complete data
            valid_scenarios.append((current_frame, valid_agents))

    if not valid_scenarios:
        return []

    # Select the scenario with the most agents
    sorted_scenarios = sorted(valid_scenarios, key=lambda x: len(x[1]), reverse=True)
    selected_scenarios = []

    selected_frames = set()
    for frame, agents in sorted_scenarios:
        # if the frame is too close to any previously selected frame, skip it
        if any(abs(frame - selected_frame) < 20 for selected_frame in selected_frames):
            continue

        # select first 8 agents
        if len(agents) > 8:
            agents = agents[:8]

        selected_scenarios.append((frame, agents))
        selected_frames.add(frame)

        if len(selected_scenarios) >= num_scenarios:
            break

    return selected_scenarios


def extract_trajectories(
    trajectories, frame, agents, frame_ids, history_len=8, future_len=12
):
    """Extract historical and future trajectories for specified agents at a given frame"""
    # Find the index of the current frame in frame_ids
    frame_idx = np.where(frame_ids == frame)[0][0]

    # Calculate historical and future frames
    history_frames = frame_ids[frame_idx - history_len + 1 : frame_idx + 1]
    future_frames = frame_ids[frame_idx + 1 : frame_idx + future_len + 1]

    # Extract trajectory data
    past = np.zeros((len(agents), history_len, 2))
    future = np.zeros((len(agents), future_len, 2))

    for i, agent_id in enumerate(agents):
        for j, hist_frame in enumerate(history_frames):
            past[i, j, 0], past[i, j, 1] = trajectories[agent_id][hist_frame]

        for j, fut_frame in enumerate(future_frames):
            future[i, j, 0], future[i, j, 1] = trajectories[agent_id][fut_frame]

    return {"past": past, "future": future, "agent_ids": agents, "time_step": frame}


def visualize_trajectory(
    data, predictions, video_path, h_path, frame, data_type, num_alternatives=10
):
    """
    Visualize trajectory data with best prediction and alternative predictions

    Args:
        data: Dictionary containing past and future trajectory data
        predictions: Numpy array of shape [num_preds, num_agents, time_steps, 2]
        num_alternatives: Number of alternative predictions to display (besides the best one)
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    # Load homography
    if h_path is None:
        H = None
    else:
        H_raw = np.loadtxt(h_path)
        H = np.linalg.inv(H_raw)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame {frame}")

    # BGR → RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb, alpha=0.8)
    image_height, image_width, _ = frame_rgb.shape
    ax.set_xlim(0, image_width)  # 예: 이미지 가로 사이즈
    ax.set_ylim(
        0, image_height
    )  # 예: 이미지 세로 사이즈 (주의: 위아래 뒤집혀 있으면 반대로)
    ax.invert_yaxis()

    image_height, image_width = frame_rgb.shape[:2]

    num_agents = data["past"].shape[0]

    # Ensure predictions has the correct dimensions
    if len(predictions.shape) == 3:  # [num_agents, time_steps, 2]
        predictions = np.expand_dims(
            predictions, axis=0
        )  # [1, num_agents, time_steps, 2]

    # If we have multiple predictions, select the best one
    if predictions.shape[0] > 1:
        # Convert to torch tensors for consistent calculation with evaluation code
        pred_tensor = torch.tensor(predictions)  # [num_preds, num_agents, time_steps, 2]
        target_tensor = torch.tensor(data["future"])  # [num_agents, time_steps, 2]

        # Calculate ADE and FDE for each prediction
        ades = []
        fdes = []
        mixed_goals = []

        for i in range(predictions.shape[0]):
            single_pred = pred_tensor[i : i + 1]  # [1, num_agents, time_steps, 2]
            ade = compute_batch_ade(single_pred, target_tensor).mean().item()
            fde = compute_batch_fde(single_pred, target_tensor).mean().item()
            mixed_goal = 0.6 * ade + 0.4 * fde
            ades.append(ade)
            fdes.append(fde)
            mixed_goals.append(mixed_goal)

        # Find the best prediction based on mixed goal (0.6*ADE + 0.4*FDE)
        best_idx = np.argmin(mixed_goals)

        # Use the best prediction
        best_prediction = predictions[best_idx]

        # Print metrics of the best prediction
        print(f"Best prediction (index {best_idx}):")
        print(f"  ADE: {ades[best_idx]:.4f}")
        print(f"  FDE: {fdes[best_idx]:.4f}")
        print(f"  Mixed Goal: {mixed_goals[best_idx]:.4f}")

        # Sort the alternatives by mixed goal (excluding the best one)
        alternative_indices = list(range(predictions.shape[0]))
        alternative_indices.remove(best_idx)
        sorted_alternatives = sorted(
            alternative_indices, key=lambda idx: mixed_goals[idx]
        )

        # Limit to the specified number of alternatives
        alternative_indices = sorted_alternatives[:num_alternatives]
    else:
        best_prediction = predictions[0]
        alternative_indices = []

    # Plot trajectories for each agent
    for i in range(num_agents):
        agent_id = int(data["agent_ids"][i])

        # Plot historical trajectory
        u_past, v_past = convert_coordinate(
            data["past"][i, :, 0], data["past"][i, :, 1], H, data_type
        )
        plt.plot(u_past, v_past, color=COLORS[0], marker="o", markersize=4, linewidth=2)

        # Plot actual future trajectory
        u_future, v_future = convert_coordinate(
            data["future"][i, :, 0], data["future"][i, :, 1], H, data_type
        )
        plt.plot(
            [u_past[-1], u_future[0]],
            [v_past[-1], v_future[0]],
            color=COLORS[1],
            linestyle="--",
            linewidth=2,
        )
        plt.plot(
            u_future, v_future, color=COLORS[1], marker="x", markersize=6, linewidth=2
        )

        # Plot best predicted trajectory
        u_best, v_best = convert_coordinate(
            best_prediction[i, :, 0], best_prediction[i, :, 1], H, data_type
        )
        plt.plot(
            [u_past[-1], u_best[0]],
            [v_past[-1], v_best[0]],
            color=COLORS[2],
            linestyle="-",
            alpha=0.8,
            linewidth=1.5,
        )
        plt.plot(
            u_best,
            v_best,
            color=COLORS[2],
            linestyle="-",
            marker="*",
            alpha=0.8,
            markersize=6,
            linewidth=1.5,
        )

        # Plot alternative predictions as dotted lines with reduced alpha
        for alt_idx in alternative_indices:
            u_pred, v_pred = convert_coordinate(
                predictions[alt_idx, i, :, 0], predictions[alt_idx, i, :, 1], H, data_type
            )
            plt.plot(
                [u_past[-1], u_pred[0]],
                [v_past[-1], v_pred[0]],
                color=COLORS[2],
                linestyle=":",
                alpha=0.3,
                linewidth=1,
            )
            plt.plot(
                u_pred, v_pred, color=COLORS[2], linestyle=":", alpha=0.3, linewidth=1
            )

    # Add legend 1
    history_line = mlines.Line2D(
        [],
        [],
        color=COLORS[0],
        marker="o",
        markersize=6,
        linewidth=2,
        label="Historical Trajectory",
    )
    actual_line = mlines.Line2D(
        [],
        [],
        color=COLORS[1],
        linestyle="--",
        marker="x",
        markersize=6,
        linewidth=2,
        label="Ground Truth",
    )
    pred_line = mlines.Line2D(
        [],
        [],
        color=COLORS[2],
        linestyle="-",
        marker="*",
        markersize=8,
        linewidth=1.5,
        alpha=1.0,
        label="Best Predicted Trajectory",
    )
    alt_line = mlines.Line2D(
        [],
        [],
        color=COLORS[2],
        linestyle=":",
        linewidth=2,
        alpha=1.0,
        label="Alternative Predictions",
    )
    plt.legend(
        handles=[history_line, actual_line, pred_line, alt_line],
        loc="upper right",
        title="Trajectory Type",
    )

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(
        "v_img.png", bbox_inches="tight", pad_inches=0, dpi=100
    )  # Use this to save the image
    plt.show()
    plt.close("all")


def process_multiple_scenarios(
    trajectories, frame_ids, scenarios, video_path, h_path, data_type, num_alternatives=5
):
    """
    Process multiple scenarios and visualize the best and alternative predictions

    Args:
        trajectories: Dictionary containing trajectory data
        frame_ids: Array of frame IDs
        scenarios: List of tuples (frame, agents) to process
        num_alternatives: Number of alternative predictions to display
    """
    for i, (frame, agents) in enumerate(scenarios):
        print(
            f"\nScenario {i+1}/{len(scenarios)} - Frame {frame}, {len(agents)} pedestrians"
        )

        data = extract_trajectories(trajectories, frame, agents, frame_ids)

        # Assuming predict_trajectory returns multiple predictions [num_preds, num_agents, time_steps, 2]
        predictions = predict_trajectory(data["past"])

        print(f"Prediction shape: {predictions.shape}")
        print(f"Scenario {i+1} - Frame {frame}:")
        visualize_trajectory(
            data, predictions, video_path, h_path, frame, data_type, num_alternatives
        )


def main():

    #### ETH
    data_type = "sdd"

    if data_type == "eth":
        data_file = "trajectory_prediction/datasets/eth/test/biwi_eth.txt"
        video_path = "video_data/eth/seq_eth.avi"
        h_path = "video_data/eth/H.txt"

    elif data_type == "hotel":
        data_file = "trajectory_prediction/datasets/hotel/test/biwi_hotel.txt"
        video_path = "video_data/hotel/seq_hotel.avi"
        h_path = "video_data/hotel/H.txt"

    elif data_type == "univ":
        data_file = "trajectory_prediction/datasets/univ/test/students003.txt"
        video_path = "video_data/univ/students003.avi"
        h_path = "video_data/univ/H.txt"

    elif data_type == "zara1":
        data_file = "trajectory_prediction/datasets/zara1/test/crowds_zara01.txt"
        video_path = "video_data/zara1/crowds_zara01.avi"
        h_path = "video_data/zara1/H.txt"

    elif data_type == "zara2":
        data_file = "trajectory_prediction/datasets/zara2/test/crowds_zara02.txt"
        video_path = "video_data/zara2/crowds_zara02.avi"
        h_path = "video_data/zara2/H.txt"

    elif data_type == "sdd":

        ## bookstore
        data_file = "trajectory_prediction/datasets/sdd/test/bookstore_0.txt"
        video_path = "video_data/sdd/bookstore/video3/video.mp4"
        h_path = None

        ## coupa
        data_file = "trajectory_prediction/datasets/sdd/test/coupa_3.txt"
        video_path = "video_data/sdd/coupa/video3/video.mp4"
        h_path = None

    trajectories, frame_ids, _ = load_data(data_file)
    scenarios = find_multiple_valid_scenarios(trajectories, frame_ids, num_scenarios=1)
    process_multiple_scenarios(
        trajectories, frame_ids, scenarios, video_path, h_path, data_type
    )


if __name__ == "__main__":
    main()
