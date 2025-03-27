import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
from tqdm import tqdm

from diffusion_process import ConditionalDiffusionProcess
from diffusion_net import ConditionalUNet1D  # 你之前写的UNet1D模型


class ConditionalTrainer:
    def __init__(self, input_length, cond_dim, timesteps,
                 learning_rate=0.001, epochs=1000, batch_size=32, name="run"):

        self.input_length = input_length
        self.cond_dim = cond_dim
        self.timesteps = timesteps
        self.epochs = epochs
        self.batch_size = batch_size
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.name = name

        self.save_dir = f"checkpoints/checkpoints_{self.name}_{self.timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        # 初始化模型（num_channels=4 对应 CIR 4个特征）
        self.model = ConditionalUNet1D(
            input_length=input_length,
            cond_dim=cond_dim,
            num_channels=4
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        self.diffusion_process = ConditionalDiffusionProcess(
            timesteps=timesteps,
            device=self.device
        )

        # 初始化 wandb
        wandb.init(project="conditional_diffusion_2", name=self.name, config={
            "input_length": input_length,
            "cond_dim": cond_dim,
            "timesteps": timesteps,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size
        })
        wandb.watch(self.model, log="all")

    def train_dataloader(self, train_loader, val_loader=None):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for coord, cir in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                x_0 = cir.permute(0, 2, 1).to(self.device)   # [B, 4, 20]
                cond = coord.to(self.device)                 # [B, 3]
                batch_size = x_0.size(0)

                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                noise = torch.randn_like(x_0)
                x_t = self.diffusion_process.get_x_t(x_0, t, noise)

                predicted_noise = self.model(x_t, cond, t)
                loss = F.mse_loss(predicted_noise, noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            wandb.log({"loss/train": avg_train_loss, "epoch": epoch})

            # 验证部分（可选）
            avg_val_loss = None
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for coord, cir in val_loader:
                        x_0 = cir.permute(0, 2, 1).to(self.device)
                        cond = coord.to(self.device)
                        batch_size = x_0.size(0)

                        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                        noise = torch.randn_like(x_0)
                        x_t = self.diffusion_process.get_x_t(x_0, t, noise)

                        predicted_noise = self.model(x_t, cond, t)
                        val_loss += F.mse_loss(predicted_noise, noise).item()

                avg_val_loss = val_loss / len(val_loader)
                wandb.log({"loss/val": avg_val_loss})

            # 保存模型
            if epoch % 100 == 0 or epoch == self.epochs - 1:
                model_save_path = os.path.join(self.save_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), model_save_path)
                print(f"[INFO] 模型已保存至 {model_save_path}")

            # 日志打印
            log_msg = f"[Epoch {epoch + 1}/{self.epochs}] Train Loss: {avg_train_loss:.6f}"
            if avg_val_loss is not None:
                log_msg += f" | Val Loss: {avg_val_loss:.6f}"
            print(log_msg)

        wandb.finish()
