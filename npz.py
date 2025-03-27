import h5py
import numpy as np

# === 设置你的数据集路径（修改为你自己的文件路径） ===
h5_path = "data/cir_data_3.27.h5"
save_path = "stats/coord_stats.npz"  # 保存路径，可以自定义

# 打开 HDF5 文件，提取坐标
with h5py.File(h5_path, 'r') as f:
    points_group = f['points']
    pids = list(points_group.keys())

    all_coords = []
    for pid in pids:
        coord = points_group[pid]['coord'][:]
        all_coords.append(coord)

    all_coords = np.array(all_coords)
    coord_mean = np.mean(all_coords, axis=0)
    coord_std = np.std(all_coords, axis=0) + 1e-6  # 防止除0

# 打印结果
print("coord_mean:", coord_mean)
print("coord_std :", coord_std)

# 保存为 .npz 文件
np.savez(save_path, mean=coord_mean, std=coord_std)
print(f"保存成功！路径: {save_path}")
