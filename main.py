import torch
from trainer import ConditionalTrainer
from torch.utils.data import DataLoader, random_split
from cir_dataset import CIRDataset


def main():
    # 超参数
    wandb_name = "run_2"
    timesteps = 1000
    input_length = 20         # ✅ CIR 的时间长度（即 cir 的序列长度）
    cond_dim = 3              # ✅ 坐标维度
    batch_size = 32
    epochs = 1000
    lr = 0.001

    config = {
        "timesteps": timesteps,
        "input_length": input_length,
        "cond_dim": cond_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "wandb_name": wandb_name
    }

    print("\n📦 Configuration:")
    for k, v in config.items():
        print(f"{k:>15}: {v}")

    # ✅ 加载 CIR 数据集（默认归一化，shape: coord [3], cir [20, 4]）
    dataset = CIRDataset('data/cir_data_3.27.h5', normalize=True)

    # 划分训练 / 验证集
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ✅ 初始化 Trainer（模型结构会自动匹配 CIR shape）
    trainer = ConditionalTrainer(
        input_length=input_length,
        cond_dim=cond_dim,
        timesteps=timesteps,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        name=wandb_name
    )

    # ✅ 开始训练
    trainer.train_dataloader(train_loader, val_loader)


if __name__ == "__main__":
    main()
