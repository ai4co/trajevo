# Define baseline results for all datasets
BASELINE_RESULTS = {
    "ETH": {
        "Social-LSTM": {"ADE": 1.09, "FDE": 2.35},
        "Social-GAN": {"ADE": 0.87, "FDE": 1.62},
        "STGAT": {"ADE": 0.65, "FDE": 1.12},
        "Social-STGCNN": {"ADE": 0.64, "FDE": 1.11},
        "PECNet": {"ADE": 0.61, "FDE": 1.07},
        "Trajectron++": {"ADE": 0.61, "FDE": 1.03},
        "SGCN": {"ADE": 0.57, "FDE": 1.00},
        "LB-EBM": {"ADE": 0.60, "FDE": 1.06},
        "AgentFormer": {"ADE": 0.46, "FDE": 0.80},
        "ExpertTraj": {"ADE": 0.37, "FDE": 0.65},
        "MemoNet": {"ADE": 0.40, "FDE": 0.61},
        "Social-Implicit": {"ADE": 0.66, "FDE": 1.44},
        "MID": {"ADE": 0.39, "FDE": 0.66},
        "NCE-STGCNN": {"ADE": 0.67, "FDE": 1.22},
        "Causal-STGCNN": {"ADE": 0.64, "FDE": 1.00},
        "GP-Graph-STGCNN": {"ADE": 0.48, "FDE": 0.77},
        "NPSN-STGCNN": {"ADE": 0.44, "FDE": 0.65},
        "EigenTrajectory-STGCNN": {"ADE": 0.36, "FDE": 0.58},
        "EigenTrajectory-AgentFormer": {"ADE": 0.36, "FDE": 0.57},
        "EigenTrajectory-Implicit": {"ADE": 0.36, "FDE": 0.57},
        "EigenTrajectory-SGCN": {"ADE": 0.36, "FDE": 0.57},
        "EigenTrajectory-PECNet": {"ADE": 0.36, "FDE": 0.57},
        "EigenTrajectory-LB-EBM": {"ADE": 0.36, "FDE": 0.53},
    },
    "HOTEL": {
        "Social-LSTM": {"ADE": 0.79, "FDE": 1.76},
        "Social-GAN": {"ADE": 0.67, "FDE": 1.37},
        "STGAT": {"ADE": 0.35, "FDE": 0.66},
        "Social-STGCNN": {"ADE": 0.49, "FDE": 0.85},
        "PECNet": {"ADE": 0.22, "FDE": 0.39},
        "Trajectron++": {"ADE": 0.20, "FDE": 0.28},
        "SGCN": {"ADE": 0.31, "FDE": 0.53},
        "LB-EBM": {"ADE": 0.21, "FDE": 0.38},
        "AgentFormer": {"ADE": 0.14, "FDE": 0.22},
        "ExpertTraj": {"ADE": 0.11, "FDE": 0.15},
        "MemoNet": {"ADE": 0.11, "FDE": 0.17},
        "Social-Implicit": {"ADE": 0.20, "FDE": 0.36},
        "MID": {"ADE": 0.13, "FDE": 0.22},
        "NCE-STGCNN": {"ADE": 0.44, "FDE": 0.68},
        "Causal-STGCNN": {"ADE": 0.38, "FDE": 0.45},
        "GP-Graph-STGCNN": {"ADE": 0.24, "FDE": 0.40},
        "NPSN-STGCNN": {"ADE": 0.21, "FDE": 0.34},
        "EigenTrajectory-STGCNN": {"ADE": 0.15, "FDE": 0.22},
        "EigenTrajectory-AgentFormer": {"ADE": 0.15, "FDE": 0.22},
        "EigenTrajectory-Implicit": {"ADE": 0.13, "FDE": 0.21},
        "EigenTrajectory-SGCN": {"ADE": 0.13, "FDE": 0.21},
        "EigenTrajectory-PECNet": {"ADE": 0.13, "FDE": 0.21},
        "EigenTrajectory-LB-EBM": {"ADE": 0.12, "FDE": 0.19},
    },
    "UNIV": {
        "Social-LSTM": {"ADE": 0.67, "FDE": 1.40},
        "Social-GAN": {"ADE": 0.76, "FDE": 1.52},
        "STGAT": {"ADE": 0.52, "FDE": 1.10},
        "Social-STGCNN": {"ADE": 0.44, "FDE": 0.79},
        "PECNet": {"ADE": 0.34, "FDE": 0.56},
        "Trajectron++": {"ADE": 0.30, "FDE": 0.55},
        "SGCN": {"ADE": 0.37, "FDE": 0.67},
        "LB-EBM": {"ADE": 0.28, "FDE": 0.54},
        "AgentFormer": {"ADE": 0.25, "FDE": 0.45},
        "ExpertTraj": {"ADE": 0.20, "FDE": 0.44},
        "MemoNet": {"ADE": 0.24, "FDE": 0.43},
        "Social-Implicit": {"ADE": 0.31, "FDE": 0.60},
        "MID": {"ADE": 0.22, "FDE": 0.45},
        "NCE-STGCNN": {"ADE": 0.47, "FDE": 0.88},
        "Causal-STGCNN": {"ADE": 0.49, "FDE": 0.81},
        "GP-Graph-STGCNN": {"ADE": 0.29, "FDE": 0.47},
        "NPSN-STGCNN": {"ADE": 0.28, "FDE": 0.44},
        "EigenTrajectory-STGCNN": {"ADE": 0.25, "FDE": 0.43},
        "EigenTrajectory-AgentFormer": {"ADE": 0.24, "FDE": 0.43},
        "EigenTrajectory-Implicit": {"ADE": 0.24, "FDE": 0.43},
        "EigenTrajectory-SGCN": {"ADE": 0.24, "FDE": 0.43},
        "EigenTrajectory-PECNet": {"ADE": 0.24, "FDE": 0.43},
        "EigenTrajectory-LB-EBM": {"ADE": 0.24, "FDE": 0.43},
    },
    "ZARA1": {
        "Social-LSTM": {"ADE": 0.47, "FDE": 1.00},
        "Social-GAN": {"ADE": 0.35, "FDE": 0.68},
        "STGAT": {"ADE": 0.34, "FDE": 0.69},
        "Social-STGCNN": {"ADE": 0.34, "FDE": 0.53},
        "PECNet": {"ADE": 0.25, "FDE": 0.45},
        "Trajectron++": {"ADE": 0.24, "FDE": 0.41},
        "SGCN": {"ADE": 0.29, "FDE": 0.51},
        "LB-EBM": {"ADE": 0.21, "FDE": 0.39},
        "AgentFormer": {"ADE": 0.18, "FDE": 0.30},
        "ExpertTraj": {"ADE": 0.15, "FDE": 0.31},
        "MemoNet": {"ADE": 0.18, "FDE": 0.32},
        "Social-Implicit": {"ADE": 0.25, "FDE": 0.50},
        "MID": {"ADE": 0.17, "FDE": 0.30},
        "NCE-STGCNN": {"ADE": 0.33, "FDE": 0.52},
        "Causal-STGCNN": {"ADE": 0.34, "FDE": 0.53},
        "GP-Graph-STGCNN": {"ADE": 0.24, "FDE": 0.40},
        "NPSN-STGCNN": {"ADE": 0.25, "FDE": 0.43},
        "EigenTrajectory-STGCNN": {"ADE": 0.22, "FDE": 0.39},
        "EigenTrajectory-AgentFormer": {"ADE": 0.22, "FDE": 0.40},
        "EigenTrajectory-Implicit": {"ADE": 0.21, "FDE": 0.37},
        "EigenTrajectory-SGCN": {"ADE": 0.20, "FDE": 0.35},
        "EigenTrajectory-PECNet": {"ADE": 0.19, "FDE": 0.35},
        "EigenTrajectory-LB-EBM": {"ADE": 0.19, "FDE": 0.33},
    },
    "ZARA2": {
        "Social-LSTM": {"ADE": 0.56, "FDE": 1.17},
        "Social-GAN": {"ADE": 0.42, "FDE": 0.84},
        "STGAT": {"ADE": 0.29, "FDE": 0.60},
        "Social-STGCNN": {"ADE": 0.30, "FDE": 0.48},
        "PECNet": {"ADE": 0.19, "FDE": 0.33},
        "Trajectron++": {"ADE": 0.18, "FDE": 0.32},
        "SGCN": {"ADE": 0.22, "FDE": 0.42},
        "LB-EBM": {"ADE": 0.15, "FDE": 0.30},
        "AgentFormer": {"ADE": 0.14, "FDE": 0.24},
        "ExpertTraj": {"ADE": 0.12, "FDE": 0.26},
        "MemoNet": {"ADE": 0.14, "FDE": 0.24},
        "Social-Implicit": {"ADE": 0.22, "FDE": 0.43},
        "MID": {"ADE": 0.13, "FDE": 0.27},
        "NCE-STGCNN": {"ADE": 0.29, "FDE": 0.48},
        "Causal-STGCNN": {"ADE": 0.32, "FDE": 0.49},
        "GP-Graph-STGCNN": {"ADE": 0.23, "FDE": 0.40},
        "NPSN-STGCNN": {"ADE": 0.22, "FDE": 0.38},
        "EigenTrajectory-STGCNN": {"ADE": 0.17, "FDE": 0.29},
        "EigenTrajectory-AgentFormer": {"ADE": 0.17, "FDE": 0.29},
        "EigenTrajectory-Implicit": {"ADE": 0.15, "FDE": 0.26},
        "EigenTrajectory-SGCN": {"ADE": 0.15, "FDE": 0.26},
        "EigenTrajectory-PECNet": {"ADE": 0.14, "FDE": 0.25},
        "EigenTrajectory-LB-EBM": {"ADE": 0.14, "FDE": 0.24},
    },
}

# Precomputed average results across all datasets
BASELINE_RESULTS_AVG = {
    "Social-LSTM": {"ADE": 0.72, "FDE": 1.54},
    "Social-GAN": {"ADE": 0.61, "FDE": 1.21},
    "STGAT": {"ADE": 0.43, "FDE": 0.83},
    "Social-STGCNN": {"ADE": 0.44, "FDE": 0.75},
    "PECNet": {"ADE": 0.32, "FDE": 0.56},
    "Trajectron++": {"ADE": 0.31, "FDE": 0.52},
    "SGCN": {"ADE": 0.35, "FDE": 0.63},
    "LB-EBM": {"ADE": 0.29, "FDE": 0.53},
    "AgentFormer": {"ADE": 0.23, "FDE": 0.40},
    "ExpertTraj": {"ADE": 0.19, "FDE": 0.36},
    "MemoNet": {"ADE": 0.21, "FDE": 0.35},
    "Social-Implicit": {"ADE": 0.33, "FDE": 0.67},
    "MID": {"ADE": 0.21, "FDE": 0.38},
    "NCE-STGCNN": {"ADE": 0.44, "FDE": 0.76},
    "Causal-STGCNN": {"ADE": 0.43, "FDE": 0.66},
    "GP-Graph-STGCNN": {"ADE": 0.29, "FDE": 0.49},
    "NPSN-STGCNN": {"ADE": 0.28, "FDE": 0.45},
    "EigenTrajectory-STGCNN": {"ADE": 0.23, "FDE": 0.38},
    "EigenTrajectory-AgentFormer": {"ADE": 0.23, "FDE": 0.38},
    "EigenTrajectory-Implicit": {"ADE": 0.22, "FDE": 0.37},
    "EigenTrajectory-SGCN": {"ADE": 0.22, "FDE": 0.36},
    "EigenTrajectory-PECNet": {"ADE": 0.22, "FDE": 0.36},
    "EigenTrajectory-LB-EBM": {"ADE": 0.21, "FDE": 0.34},
}


def compare_baselines(ade, fde, dataset="ETH"):
    """
    Compare the baseline results with the given ADE and FDE values for a specific dataset.

    Args:
        ade (float): Average Displacement Error of your model
        fde (float): Final Displacement Error of your model
        dataset (str): The dataset to compare with. Options: "ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2", "AVG"

    Returns:
        None: Prints a formatted comparison table
    """
    if dataset == "AVG":
        baseline_results = BASELINE_RESULTS_AVG
    else:
        if dataset not in BASELINE_RESULTS:
            raise ValueError(
                f"Dataset {dataset} not found. Choose from: ETH, HOTEL, UNIV, ZARA1, ZARA2, AVG"
            )
        baseline_results = BASELINE_RESULTS[dataset]

    # Find the best (minimum) ADE and FDE values from baselines
    best_ade = min(model["ADE"] for model in baseline_results.values())
    best_fde = min(model["FDE"] for model in baseline_results.values())

    # Print header
    print(f"Dataset: {dataset}")
    print(f"{'Model Name':<30}\t{'ADE':>6}\t{'ADE Gap':>10}\t{'FDE':>6}\t{'FDE Gap':>10}")
    print("-" * 75)

    # Print each baseline model's results in original order
    for model_name, metrics in baseline_results.items():
        # Calculate gaps relative to best values
        ade_gap = ((metrics["ADE"] - best_ade) / best_ade) * 100
        fde_gap = ((metrics["FDE"] - best_fde) / best_fde) * 100

        # Format metrics
        ade_val = f"{metrics['ADE']:.2f}"
        fde_val = f"{metrics['FDE']:.2f}"
        ade_gap_str = f"{ade_gap:+.2f}%"
        fde_gap_str = f"{fde_gap:+.2f}%"

        # Print row with formatting
        print(
            f"{model_name:<30}\t{ade_val:>6}\t{ade_gap_str:>10}\t{fde_val:>6}\t{fde_gap_str:>10}"
        )

    # Print separator
    print("-" * 75)

    # Calculate gaps for your model
    ade_gap = ((ade - best_ade) / best_ade) * 100
    fde_gap = ((fde - best_fde) / best_fde) * 100

    # Print your model's results
    ade_str = f"{ade:.2f}"
    fde_str = f"{fde:.2f}"
    ade_gap_str = f"{ade_gap:+.2f}%"
    fde_gap_str = f"{fde_gap:+.2f}%"
    print(
        f"{'Our':<30}\t{ade_str:>6}\t{ade_gap_str:>10}\t{fde_str:>6}\t{fde_gap_str:>10}"
    )


def compare_all_datasets(our_results):
    """
    Compare our model's results with baselines across all datasets.

    Args:
        our_results (dict): Dictionary with keys as dataset names and values as {"ADE": float, "FDE": float}
            Example: {"ETH": {"ADE": 0.35, "FDE": 0.52}, "HOTEL": {"ADE": 0.10, "FDE": 0.14}, ...}

    Returns:
        None: Prints formatted comparison tables for each dataset
    """
    for dataset in our_results:
        if dataset not in BASELINE_RESULTS and dataset != "AVG":
            print(f"Warning: Dataset {dataset} not found in baseline results. Skipping.")
            continue

        print("\n" + "=" * 75)
        compare_baselines(
            our_results[dataset]["ADE"], our_results[dataset]["FDE"], dataset
        )


# Example usage:
if __name__ == "__main__":
    # Example 1: Compare on a single dataset
    # compare_baselines(0.35, 0.50, "ETH")

    # Example 2: Compare across all datasets
    our_results = {
        "ETH": {"ADE": 0.35, "FDE": 0.50},
        "HOTEL": {"ADE": 0.10, "FDE": 0.14},
        "UNIV": {"ADE": 0.23, "FDE": 0.41},
        "ZARA1": {"ADE": 0.18, "FDE": 0.30},
        "ZARA2": {"ADE": 0.13, "FDE": 0.23},
        "AVG": {"ADE": 0.20, "FDE": 0.32},
    }
    compare_all_datasets(our_results)
