import torch
import numpy as np
from cir_dataset import CIRDataset
from inference import ConditionalInference


def load_normalization_stats(stats_path):
    stats = np.load(stats_path)
    mean = stats["mean"]
    std = stats["std"]
    return mean, std


def standardize_coord(coord_np, mean, std):
    return (coord_np - mean) / std


def destandardize_cir(cir_tensor, cir_mean, cir_std):
    """
    cir_tensor: [B, 20, 4]，其中前两列标准化过
    cir_mean, cir_std: shape (3,)，仅用于 delay 和 amplitude
    """
    delay = cir_tensor[..., 0] * cir_std[0] + cir_mean[0]
    amplitude = cir_tensor[..., 1] * cir_std[1] + cir_mean[1]
    phase_cos = cir_tensor[..., 2]
    phase_sin = cir_tensor[..., 3]
    phase = torch.atan2(phase_sin, phase_cos)

    cir_destandardized = torch.stack([delay, amplitude, phase], dim=-1)  # [B, 20, 3]
    return cir_destandardized


def find_closest_real_cir(real_coord, dataset):
    """
    real_coord: shape (3,)
    dataset: CIRDataset
    返回最接近坐标对应的真实 CIR
    """
    real_coord = torch.tensor(real_coord, dtype=torch.float32)
    all_coords = []
    for i in range(len(dataset)):
        coord_i, _ = dataset[i]
        all_coords.append(coord_i * torch.tensor(dataset.coord_std) + torch.tensor(dataset.coord_mean))
    all_coords = torch.stack(all_coords)

    dists = torch.norm(all_coords - real_coord, dim=1)
    idx = torch.argmin(dists)
    coord_match, cir_match = dataset[idx]
    return coord_match, cir_match


def generate_from_real_coord(
        real_coord_np, model_path, stats_path, cir_dataset_path, timesteps=100, input_length=20, cond_dim=3
):
    """
    real_coord_np: (3,) numpy array，真实坐标
    返回：
        - 生成 CIR: [20, 3]
        - 最接近的真实 CIR: [20, 3]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载标准化参数
    coord_mean, coord_std = load_normalization_stats(stats_path)

    # 加载数据集，用于取 CIR mean/std + 找最接近的真实样本
    dataset = CIRDataset(cir_dataset_path, normalize=True)
    cir_mean = dataset.cir_mean
    cir_std = dataset.cir_std

    # 坐标标准化
    coord_norm = standardize_coord(real_coord_np, coord_mean, coord_std)
    coord_tensor = torch.tensor(coord_norm, dtype=torch.float32).unsqueeze(0)  # [1, 3]

    # 初始化推理引擎
    engine = ConditionalInference(
        model_path=model_path,
        input_length=input_length,
        cond_dim=cond_dim,
        timesteps=timesteps,
        device=device
    )

    # 推理
    cir_generated = engine.generate(coord_tensor, batch_size=1)  # [1, 20, 4]
    cir_generated = cir_generated[0]  # 去掉 batch 维

    # 反标准化生成 CIR
    cir_gen_destd = destandardize_cir(cir_generated, cir_mean, cir_std)  # [20, 3]

    # 找最近坐标的真实 CIR
    coord_match, cir_real = find_closest_real_cir(real_coord_np, dataset)
    cir_real_destd = destandardize_cir(cir_real.unsqueeze(0), cir_mean, cir_std)[0]  # [20, 3]

    return cir_gen_destd.cpu().numpy(), cir_real_destd.cpu().numpy()
