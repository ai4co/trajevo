import math
import os

import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler


def compute_batch_ade_ret(pred, gt):
    """NOTE: is ADE correct, i.e. should be avg across all agents?"""
    temp = (pred - gt).norm(p=2, dim=-1)
    ADEs, idx = temp.mean(dim=[1, 2]).min(
        dim=0
    )  # NOTE: this should be avg across all agents??
    ade_all = temp.mean(dim=[1, 2])
    return ADEs, idx, ade_all


def compute_batch_fde_ret(pred, gt):
    temp = (pred - gt).norm(p=2, dim=-1)
    FDEs, idx = temp[:, :, -1].mean(dim=[-1]).min(dim=0)
    fde_all = temp[:, :, -1].mean(dim=[-1])
    return FDEs, idx, fde_all


# def compute_diversity_eval(predictions):
#     """compute diversity of trajectories

#     Args:
#         predictions: shape of [num_samples, num_agents, seq_len, 2]
#     """
#     try:
#         # Convert PyTorch tensor to NumPy array
#         if isinstance(predictions, torch.Tensor):
#             predictions = predictions.detach().cpu().numpy()

#         # ensure enough samples to compute diversity
#         if predictions.shape[0] < 2:
#             return 0.0

#         # compute distance matrix between samples (avoid empty slices)
#         distances = []
#         for i in range(predictions.shape[0]):
#             for j in range(i + 1, predictions.shape[0]):
#                 # compute average euclidean distance between pairs of trajectories
#                 dist = np.sqrt(np.sum((predictions[i] - predictions[j]) ** 2, axis=-1))
#                 # ensure non-empty array for safe averaging
#                 if dist.size > 0:
#                     distances.append(np.mean(dist))

#         # ensure non-empty array for safe averaging
#         if len(distances) > 0:
#             spatial_diversity = np.mean(distances)
#         else:
#             spatial_diversity = 0.0

#         # compute direction diversity (add safety check)
#         direction_diversity = 0.0
#         if predictions.shape[1] > 0 and predictions.shape[2] > 1:
#             # compute direction change for each trajectory
#             directions = []
#             for sample_idx in range(predictions.shape[0]):
#                 # trajectory direction change
#                 diffs = predictions[sample_idx, :, 1:] - predictions[sample_idx, :, :-1]
#                 if diffs.size > 0:
#                     # avoid zero vector
#                     norms = np.maximum(np.linalg.norm(diffs, axis=-1), 1e-6)
#                     # compute standard deviation of normalized direction changes
#                     dir_std = np.std(norms) if norms.size > 0 else 0
#                     directions.append(dir_std)

#             if len(directions) > 0:
#                 direction_diversity = np.mean(directions)

#         # combine diversity scores
#         diversity = spatial_diversity + direction_diversity
#         return max(0.0, diversity)  # ensure non-negative

#     except Exception as e:
#         print(f"error in computing diversity: {e}")
#         return 0.0  # return safe value when error occurs


# def compute_batch_ade(pred, gt):
#     """Compute ADE(average displacement error) scores for each pedestrian
#     Args:
#         pred (np.ndarray): (num_samples, num_ped, seq_len, 2)
#         gt (np.ndarray): (1, num_ped, seq_len, 2) or (num_ped, seq_len, 2)
#     Returns:
#         ADEs (np.ndarray): (num_ped,)
#     """
#     # Compute L2 norm along last dimension (equivalent to torch.norm with p=2)
#     temp = np.sqrt(np.sum((pred - gt)**2, axis=-1))

#     # Mean along sequence length dimension
#     temp_mean = np.mean(temp, axis=2)

#     # Min along sample dimension (equivalent to torch.min with dim=0)
#     ADEs = np.min(temp_mean, axis=0)

#     return ADEs


# def compute_batch_fde(pred, gt):
#     """Compute FDE(final displacement error) scores for each pedestrian
#     Args:
#         pred (np.ndarray): (num_samples, num_ped, seq_len, 2)
#         gt (np.ndarray): (1, num_ped, seq_len, 2) or (num_ped, seq_len, 2)
#     Returns:
#         FDEs (np.ndarray): (num_ped,)
#     """
#     # Compute L2 norm along last dimension
#     temp = np.sqrt(np.sum((pred - gt)**2, axis=-1))

#     # Get the last time step and min across samples
#     FDEs = np.min(temp[:, :, -1], axis=0)

#     return FDEs


# def compute_batch_tcc(pred, gt):
#     """Compute TCC(temporal correlation coefficient) scores for each pedestrian
#     Args:
#         pred (np.ndarray): (num_samples, num_ped, seq_len, 2)
#         gt (np.ndarray): (1, num_ped, seq_len, 2) or (num_ped, seq_len, 2)
#     Returns:
#         TCCs (np.ndarray): (num_ped,)
#     """
#     # Squeeze the first dimension if gt has 4 dimensions
#     if gt.ndim == 4:
#         gt = np.squeeze(gt, axis=0)

#     # Compute L2 norm along last dimension
#     temp = np.sqrt(np.sum((pred - gt)**2, axis=-1))

#     # Find the indices of the best predictions based on final displacement
#     best_indices = np.argmin(temp[:, :, -1], axis=0)

#     # Select the best predictions for each pedestrian
#     pred_best = pred[best_indices, np.arange(pred.shape[1]), :, :]

#     # Stack the best predictions and ground truth
#     pred_gt_stack = np.stack([pred_best, gt], axis=0)

#     # Permute dimensions (equivalent to torch.permute)
#     pred_gt_stack = np.transpose(pred_gt_stack, (3, 1, 0, 2))

#     # Compute covariance
#     mean = np.mean(pred_gt_stack, axis=-1, keepdims=True)
#     covariance = pred_gt_stack - mean
#     factor = 1 / (covariance.shape[-1] - 1)
#     covariance = factor * np.matmul(covariance, np.transpose(covariance, (0, 1, 3, 2)))

#     # Extract the diagonal elements (variances)
#     variance = np.diagonal(covariance, axis1=-2, axis2=-1)

#     # Compute standard deviation
#     stddev = np.sqrt(variance)

#     # Compute correlation coefficient
#     corrcoef = covariance / np.expand_dims(stddev, -1) / np.expand_dims(stddev, -2)

#     # Clamp values to [-1, 1]
#     corrcoef = np.clip(corrcoef, -1, 1)

#     # Replace NaN values with 0
#     corrcoef = np.nan_to_num(corrcoef)

#     # Compute mean TCC across dimensions
#     TCCs = np.mean(corrcoef[:, :, 0, 1], axis=0)

#     return TCCs


# def compute_batch_col(pred, gt):
#     """Compute COL(collision rate) scores for each pedestrian
#     Args:
#         pred (np.ndarray): (num_samples, num_ped, seq_len, 2)
#         gt (np.ndarray): (1, num_ped, seq_len, 2) or (num_ped, seq_len, 2)
#     Returns:
#         COLs (np.ndarray): (num_ped,)
#     """
#     # Permute dimensions (equivalent to torch.permute)
#     pred = np.transpose(pred, (0, 2, 1, 3))

#     num_interp, thres = 4, 0.2

#     # Extract first point
#     pred_fp = pred[:, [0], :, :]

#     # Compute relative displacements
#     pred_rel = pred[:, 1:] - pred[:, :-1]

#     # Create dense relative displacements
#     pred_rel_dense = pred_rel / num_interp

#     # Repeat interleave operation
#     shape = pred_rel_dense.shape
#     pred_rel_dense = np.repeat(np.expand_dims(pred_rel_dense, axis=2), num_interp, axis=2)
#     pred_rel_dense = pred_rel_dense.reshape(shape[0], num_interp * (shape[1] - 1), shape[2], shape[3])

#     # Concatenate first point and cumulative sum
#     pred_dense = np.concatenate([pred_fp, pred_rel_dense], axis=1)
#     pred_dense = np.cumsum(pred_dense, axis=1)

#     # Create collision mask
#     col_mask = pred_dense[:, :3 * num_interp + 2]
#     col_mask = np.repeat(np.expand_dims(col_mask, axis=2), pred.shape[2], axis=2)

#     # Compute pairwise distances
#     col_mask_diff = col_mask - np.transpose(col_mask, (0, 1, 3, 2))
#     col_mask_dist = np.sqrt(np.sum(col_mask_diff**2, axis=-1))

#     # Add identity matrix to avoid self-collisions
#     eye_matrix = np.eye(pred.shape[2])[np.newaxis, np.newaxis, :, :]
#     col_mask_dist = col_mask_dist + eye_matrix

#     # Find minimum distances and check threshold
#     col_mask = np.min(col_mask_dist, axis=1) < thres

#     # Compute collision rate
#     COLs = np.mean(np.any(col_mask, axis=1).astype(float), axis=0) * 100

#     return COLs


def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non-linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=12,
        skip=1,
        threshold=0.02,
        min_ped=1,
        delim="\t",
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non-linear traj when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a sequence
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx : idx + self.seq_len], axis=0
                )
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        self.num_peds_in_seq = np.array(num_peds_in_seq)

        # Convert numpy -> Torch Tensor
        self.obs_traj = (
            torch.from_numpy(seq_list[:, :, : self.obs_len])
            .type(torch.float)
            .permute(0, 2, 1)
        )  # NTC
        self.pred_traj = (
            torch.from_numpy(seq_list[:, :, self.obs_len :])
            .type(torch.float)
            .permute(0, 2, 1)
        )  # NTC
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end],
            self.pred_traj[start:end],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end],
            None,
            [[0, end - start]],
        ]
        return out


class TrajBatchSampler(Sampler):
    r"""Samples batched elements by yielding a mini-batch of indices.
    Args:
        data_source (Dataset): dataset to sample from
        batch_size (int): Size of mini-batch.
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        generator (Generator): Generator used in sampling.
    """

    def __init__(
        self, data_source, batch_size=64, shuffle=False, drop_last=False, generator=None
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        assert len(self.data_source) == len(self.data_source.num_peds_in_seq)

        if self.shuffle:
            if self.generator is None:
                generator = torch.Generator()
                generator.manual_seed(
                    int(torch.empty((), dtype=torch.int64).random_().item())
                )
            else:
                generator = self.generator
            indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        else:
            indices = list(range(len(self.data_source)))
        num_peds_indices = self.data_source.num_peds_in_seq[indices]

        batch = []
        total_num_peds = 0
        for idx, num_peds in zip(indices, num_peds_indices):
            batch.append(idx)
            total_num_peds += num_peds
            if total_num_peds >= self.batch_size:
                yield batch
                batch = []
                total_num_peds = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Approximated number of batches.
        # The order of trajectories can be shuffled, so this number can vary from run to run.
        if self.drop_last:
            return sum(self.data_source.num_peds_in_seq) // self.batch_size
        else:
            return (
                sum(self.data_source.num_peds_in_seq) + self.batch_size - 1
            ) // self.batch_size


def traj_collate_fn(data):
    r"""Collate function for the dataloader

    Args:
        data (list): list of tuples of (obs_seq, pred_seq, non_linear_ped, loss_mask, seq_start_end)

    Returns:
        obs_seq_list (torch.Tensor): (num_ped, obs_len, 2)
        pred_seq_list (torch.Tensor): (num_ped, pred_len, 2)
        non_linear_ped_list (torch.Tensor): (num_ped,)
        loss_mask_list (torch.Tensor): (num_ped, obs_len + pred_len)
        scene_mask (torch.Tensor): (num_ped, num_ped)
        seq_start_end (torch.Tensor): (num_ped, 2)
    """

    obs_seq_list, pred_seq_list, non_linear_ped_list, loss_mask_list, _, _ = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]
    seq_start_end = torch.LongTensor(seq_start_end)
    scene_mask = torch.zeros(sum(_len), sum(_len), dtype=torch.bool)
    for idx, (start, end) in enumerate(seq_start_end):
        scene_mask[start:end, start:end] = 1

    out = [
        torch.cat(obs_seq_list, dim=0),
        torch.cat(pred_seq_list, dim=0),
        torch.cat(non_linear_ped_list, dim=0),
        torch.cat(loss_mask_list, dim=0),
        scene_mask,
        seq_start_end,
    ]
    return tuple(out)


def get_dataloader(data_dir, phase, obs_len, pred_len, batch_size):
    r"""Get dataloader for a specific phase

    Args:
        data_dir (str): path to the dataset directory
        phase (str): phase of the data, one of 'train', 'val', 'test'
        obs_len (int): length of observed trajectory
        pred_len (int): length of predicted trajectory
        batch_size (int): batch size

    Returns:
        loader_phase (torch.utils.data.DataLoader): dataloader for the specific phase
    """

    assert phase in ["train", "val", "test"]

    data_set = data_dir + "/" + phase + "/"
    shuffle = True if phase == "train" else False
    drop_last = True if phase == "train" else False

    dataset_phase = TrajectoryDataset(data_set, obs_len=obs_len, pred_len=pred_len)
    sampler_phase = None
    if batch_size > 1:
        sampler_phase = TrajBatchSampler(
            dataset_phase, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
    loader_phase = DataLoader(
        dataset_phase,
        collate_fn=traj_collate_fn,
        batch_sampler=sampler_phase,
        pin_memory=True,
    )
    return loader_phase


def compute_batch_ade(pred, gt):
    r"""Compute ADE(average displacement error) scores for each pedestrian

    Args:
        pred (torch.Tensor): (num_samples, num_ped, seq_len, 2)
        gt (torch.Tensor): (1, num_ped, seq_len, 2) or (num_ped, seq_len, 2)

    Returns:
        ADEs (np.ndarray): (num_ped,)
    """

    temp = (pred - gt).norm(p=2, dim=-1)
    ADEs = temp.mean(dim=2).min(dim=0)[0]
    return ADEs.detach().cpu().numpy()


def compute_batch_fde(pred, gt):
    r"""Compute FDE(final displacement error) scores for each pedestrian

    Args:
        pred (torch.Tensor): (num_samples, num_ped, seq_len, 2)
        gt (torch.Tensor): (1, num_ped, seq_len, 2) or (num_ped, seq_len, 2)

    Returns:
        FDEs (np.ndarray): (num_ped,)
    """

    temp = (pred - gt).norm(p=2, dim=-1)
    FDEs = temp[:, :, -1].min(dim=0)[0]
    return FDEs.detach().cpu().numpy()


# def compute_diversity(prediction):
#     """compute diversity of predicted trajectories

#     Args:
#         prediction: shape of [20, num_agents, 12, 2]

#     Returns:
#         diversity score, higher means more diverse
#     """
#     # compute average distance between 20 samples
#     distances = []
#     for i in range(len(prediction)):
#         for j in range(i + 1, len(prediction)):
#             # compute average distance between two trajectories
#             distance = np.mean(
#                 np.sqrt(np.sum((prediction[i] - prediction[j]) ** 2, axis=-1))
#             )
#             distances.append(distance)

#     # average distance as diversity metric
#     diversity = np.mean(distances)

#     # add direction diversity
#     directions = []
#     for traj in prediction:
#         # compute direction change for each time step
#         direction_changes = np.diff(traj, axis=1)
#         angle_changes = np.arctan2(direction_changes[:, :, 1], direction_changes[:, :, 0])
#         std_angles = np.std(angle_changes)
#         directions.append(std_angles)

#     # standard deviation of direction changes also as diversity metric
#     direction_diversity = np.mean(directions)

#     return diversity + direction_diversity


def compute_batch_tcc(pred, gt):
    r"""Compute TCC(temporal correlation coefficient) scores for each pedestrian

    Args:
        pred (torch.Tensor): (num_samples, num_ped, seq_len, 2)
        gt (torch.Tensor): (1, num_ped, seq_len, 2) or (num_ped, seq_len, 2)

    Returns:
        TCCs (np.ndarray): (num_ped,)
    """

    gt = gt.squeeze(dim=0) if gt.dim() == 4 else gt
    temp = (pred - gt).norm(p=2, dim=-1)
    pred_best = pred[temp[:, :, -1].argmin(dim=0), range(pred.size(1)), :, :]
    pred_gt_stack = torch.stack([pred_best, gt], dim=0)
    pred_gt_stack = pred_gt_stack.permute(3, 1, 0, 2)
    covariance = pred_gt_stack - pred_gt_stack.mean(dim=-1, keepdim=True)
    factor = 1 / (covariance.shape[-1] - 1)
    covariance = factor * covariance @ covariance.transpose(-1, -2)
    variance = covariance.diagonal(offset=0, dim1=-2, dim2=-1)
    stddev = variance.sqrt()
    corrcoef = covariance / stddev.unsqueeze(-1) / stddev.unsqueeze(-2)
    corrcoef = corrcoef.clamp(-1, 1)
    corrcoef[torch.isnan(corrcoef)] = 0
    TCCs = corrcoef[:, :, 0, 1].mean(dim=0)
    return TCCs.detach().cpu().numpy()


def compute_batch_col(pred, gt):
    r"""Compute COL(collision rate) scores for each pedestrian

    Args:
        pred (torch.Tensor): (num_samples, num_ped, seq_len, 2)
        gt (torch.Tensor): (1, num_ped, seq_len, 2) or (num_ped, seq_len, 2)

    Returns:
        COLs (np.ndarray): (num_ped,)
    """

    pred = pred.permute(0, 2, 1, 3)
    num_interp, thres = 4, 0.2
    pred_fp = pred[:, [0], :, :]
    pred_rel = pred[:, 1:] - pred[:, :-1]
    pred_rel_dense = (
        pred_rel.div(num_interp)
        .unsqueeze(dim=2)
        .repeat_interleave(repeats=num_interp, dim=2)
        .contiguous()
    )
    pred_rel_dense = pred_rel_dense.reshape(
        pred.size(0), num_interp * (pred.size(1) - 1), pred.size(2), pred.size(3)
    )
    pred_dense = torch.cat([pred_fp, pred_rel_dense], dim=1).cumsum(dim=1)
    col_mask = (
        pred_dense[:, : 3 * num_interp + 2]
        .unsqueeze(dim=2)
        .repeat_interleave(repeats=pred.size(2), dim=2)
    )
    col_mask = (col_mask - col_mask.transpose(2, 3)).norm(p=2, dim=-1)
    col_mask = (
        col_mask.add(torch.eye(n=pred.size(2), device=pred.device)[None, None, :, :])
        .min(dim=1)[0]
        .lt(thres)
    )
    COLs = col_mask.sum(dim=1).gt(0).type(pred.type()).mean(dim=0).mul(100)
    return COLs.detach().cpu().numpy()


def load_limited_data_per_scene(
    dataset_dir, phase, obs_len, pred_len, samples_per_scene=10, seed=42
):
    """
    Load limited number of samples per scene from each scene file, ensuring determinism
    """
    np.random.seed(seed)  # set random seed to ensure determinism

    data_path = os.path.join(dataset_dir, phase)
    all_files = [f for f in os.listdir(data_path) if f.endswith(".txt")]

    all_inputs = []
    all_targets = []

    print(
        f"Load data from {len(all_files)} scene files, each scene limited to {samples_per_scene} samples"
    )

    for file_name in all_files:
        scene_name = os.path.splitext(file_name)[0]
        print(f"Processing scene: {scene_name}")

        # temporarily modify folder structure, create a temporary folder containing only the current file
        temp_dir = os.path.join(dataset_dir, "temp")
        temp_phase_dir = os.path.join(temp_dir, phase)  # use original phase name
        os.makedirs(temp_phase_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_phase_dir, file_name)

        # copy file content
        with open(os.path.join(data_path, file_name), "r") as src:
            with open(temp_file_path, "w") as dst:
                dst.write(src.read())

        # create dataloader for single file
        try:
            scene_loader = get_dataloader(
                temp_dir, phase, obs_len, pred_len, batch_size=1
            )

            # collect samples of this scene
            scene_samples = []
            for batch in scene_loader:
                scene_samples.append((batch[0].numpy(), batch[1].numpy()))

            # if the number of samples is less than the required number, use all samples
            total_samples = len(scene_samples)
            samples_to_use = min(samples_per_scene, total_samples)

            # generate deterministic random indices
            if samples_to_use < total_samples:
                indices = np.random.choice(total_samples, samples_to_use, replace=False)
                indices = sorted(indices)  # sort to maintain some continuity
            else:
                indices = list(range(total_samples))

            # get selected samples
            for idx in indices:
                all_inputs.append(scene_samples[idx][0])
                all_targets.append(scene_samples[idx][1])

            print(f"Loaded {len(indices)} samples from scene {scene_name}")

        finally:
            # clean up temporary directory
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_phase_dir):
                os.rmdir(temp_phase_dir)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    print(f"Loaded {len(all_inputs)} samples in total")
    return all_inputs, all_targets
