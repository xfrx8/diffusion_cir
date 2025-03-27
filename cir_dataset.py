import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class CIRDataset(Dataset):
    def __init__(self, h5_file, normalize=True):
        super().__init__()
        self.h5_file = h5_file
        self.normalize = normalize
        self.file = h5py.File(h5_file, 'r')

        self.points_group = self.file['points']
        self.pids = list(self.points_group.keys())

        if normalize:
            self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        all_coords = []
        all_delays = []
        all_amplitudes = []

        for pid in self.pids:
            coord = self.points_group[pid]['coord'][:]
            cir = self.points_group[pid]['cir'][:]  # shape (20, 3)

            all_coords.append(coord)

            delay = cir[:, 0]
            amplitude = cir[:, 1]

            all_delays.append(delay)
            all_amplitudes.append(amplitude)

        self.coord_mean = np.mean(all_coords, axis=0)
        self.coord_std = np.std(all_coords, axis=0) + 1e-6

        all_delays = np.concatenate(all_delays, axis=0)
        all_amplitudes = np.concatenate(all_amplitudes, axis=0)

        self.delay_mean = np.mean(all_delays)
        self.delay_std = np.std(all_delays) + 1e-6

        self.amp_mean = np.mean(all_amplitudes)
        self.amp_std = np.std(all_amplitudes) + 1e-6

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        pid = self.pids[index]
        coord = self.points_group[pid]['coord'][:]     # (3,)
        cir = self.points_group[pid]['cir'][:]         # (20, 3)

        if self.normalize:
            coord = (coord - self.coord_mean) / self.coord_std

            delay = cir[:, 0]
            amplitude = cir[:, 1]
            phase = cir[:, 2]

            delay = (delay - self.delay_mean) / self.delay_std
            amplitude = (amplitude - self.amp_mean) / self.amp_std

            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)

            cir = np.stack([delay, amplitude, cos_phase, sin_phase], axis=1)
        else:
            delay = cir[:, 0]
            amplitude = cir[:, 1]
            phase = cir[:, 2]
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            cir = np.stack([delay, amplitude, cos_phase, sin_phase], axis=1)

        return torch.from_numpy(coord).float(), torch.from_numpy(cir).float()

    def close(self):
        self.file.close()

    def __del__(self):
        self.close()


# ✅ 加入 main 测试代码
if __name__ == "__main__":
    h5_path = "data/cir_data_3.27.h5"  # TODO: 替换为你自己的路径

    dataset = CIRDataset(h5_path, normalize=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for coords, cirs in dataloader:
        print("Coords shape:", coords.shape)   # (B, 3)
        print("CIRs shape:", cirs.shape)       # (B, 20, 4)
        print("Sample coord:", coords[0])
        print("Sample CIR[0]:", cirs[0])       # 20×4
        break

    dataset.close()
