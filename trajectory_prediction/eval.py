import argparse
import os

from functools import partial
from multiprocessing import Pool
from os import path

import numpy as np
import torch

from trajectory_prediction.utils import (
    compute_batch_ade_ret,
    compute_batch_fde_ret,
    load_limited_data_per_scene,
)

torch.multiprocessing.set_sharing_strategy("file_system")


NUM_TRAJECTORIES = (
    20  # number of trajectories to generate by the heuristic per test sample
)

SEED = 3


def process_trajectory(traj_idx, input_traj, target_traj, code_args=None):

    # set random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Run prediction on the trajectory
    if code_args is not None:
        gpt_prediction = predict_trajectory(input_traj, **eval(code_args))
    else:
        gpt_prediction = predict_trajectory(input_traj)

    gpt_prediction_tensor = torch.tensor(gpt_prediction)
    target_traj = torch.tensor(target_traj)

    # Compute ADE and FDE for the prediction
    _, _, ade_all = compute_batch_ade_ret(gpt_prediction_tensor, target_traj)
    _, _, fde_all = compute_batch_fde_ret(gpt_prediction_tensor, target_traj)

    assert (
        ade_all.shape[0] == NUM_TRAJECTORIES
    ), f"ade first shape should be {NUM_TRAJECTORIES}, but got {ade_all.shape[0]}"

    # Calculate MSE for all predictions
    mse_all = torch.mean((gpt_prediction_tensor - target_traj) ** 2, dim=(1, 2, 3))

    # Find the best trajectory based on MSE
    minmse, minmse_idx = torch.min(mse_all, dim=0)

    # Get the ADE and FDE for the trajectory with the minimum MSE
    minade = ade_all[minmse_idx]
    minfde = fde_all[minmse_idx]

    return {
        "traj_idx": traj_idx,
        "minade_idx": minmse_idx,
        "minfde_idx": minmse_idx,
        "minade": minade,
        "minfde": minfde,
        "mse": minmse,
    }


def main(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = path.join(current_dir, "datasets", args.dataset)
    obs_len = 8
    pred_len = 12

    # Load data
    num_samples_per_scene = (
        args.samples_per_scene if not args.test else args.samples_per_scene_test
    )
    all_inputs, all_targets = load_limited_data_per_scene(
        dataset_dir,
        "test" if args.test else "val",
        obs_len,
        pred_len,
        samples_per_scene=num_samples_per_scene,
    )

    with torch.inference_mode():
        # Prepare multiprocessing pool
        with Pool(processes=args.num_processes) as pool:
            # Prepare arguments for each trajectory
            tasks = [
                (traj_idx, input_traj, target_traj)
                for traj_idx, (input_traj, target_traj) in enumerate(
                    zip(all_inputs, all_targets)
                )
            ]

            # Map the tasks to the pool and run them in parallel
            best_per_trajectory = pool.starmap(
                partial(process_trajectory, code_args=args.code_args), tasks
            )

    # Calculate the averages for the results
    avg_best_ade = np.mean([traj_result["minade"] for traj_result in best_per_trajectory])
    avg_best_fde = np.mean([traj_result["minfde"] for traj_result in best_per_trajectory])
    avg_best_mse = np.mean(
        [traj_result["mse"] for traj_result in best_per_trajectory]
    )

    # Count the best_idx
    best_idx_counts = {i: 0 for i in range(20)}
    for traj_result in best_per_trajectory:
        best_idx = int(traj_result["minade_idx"])
        best_idx_counts[best_idx] += 1

    # Print the results
    print("<stats>")
    print("Statistics of trajectory index counts with lowest MSE.")
    print(
        "These help us understand which heuristics contribute to the performance for at least some trajectories."
    )
    print("Traj Index: Count")
    for best_idx, count in best_idx_counts.items():
        print(f"{best_idx}: {count}")
    print("</stats>")

    print("\n[*] Average metrics with per-trajectory optimal parameters:")
    print(f"ADE: {avg_best_ade:.6f}")
    print(f"FDE: {avg_best_fde:.6f}")

    # Needed for running TrajEvo
    print("\n[*] Minimum MSE:")
    print(float(avg_best_mse))


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="eth")
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Test mode. Use --no-test to disable.",
    )
    parser.add_argument(
        "--samples_per_scene", type=int, default=20, help="Number of samples per scene."
    )
    parser.add_argument(
        "--samples_per_scene_test",
        type=int,
        default=1e42,
        help="Number of samples per scene in test mode.",
    )
    parser.add_argument(
        "--code_path", type=str, default=None, help="Path to the code to evaluate."
    )
    parser.add_argument(
        "--code_function", type=str, default=None, help="Function to evaluate."
    )
    parser.add_argument(
        "--code_args", type=str, default=None, help="Arguments to pass to the code."
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=10,
        help="Number of processes to use for parallel evaluation.",
    )

    args = parser.parse_args()

    # If code path is None, then the predict_trajectory default from GPT is used
    code_args = {}
    if args.code_path is None:
        try:
            from trajectory_prediction.gpt import (
                predict_trajectory_v2 as predict_trajectory,
            )
        except ImportError:
            from trajectory_prediction.gpt import predict_trajectory
    else:
        code_path = args.code_path
        code_function = args.code_function
        code_args = args.code_args
        assert code_function is not None, "code_function must be provided"
        import importlib
        import sys

        sys.path.append(os.path.dirname(code_path))
        module_name = os.path.basename(code_path).replace(".py", "")
        predict_trajectory = getattr(importlib.import_module(module_name), code_function)
        assert callable(predict_trajectory), "code_function must be callable"
        if code_args is not None:
            code_args = eval(code_args)

    main(args)
