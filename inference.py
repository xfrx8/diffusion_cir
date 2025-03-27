import torch
import torch.nn.functional as F
from diffusion_net import ConditionalUNet1D
from diffusion_process import ConditionalDiffusionProcess


class ConditionalInference:
    def __init__(self, model_path, input_length=20, cond_dim=3, timesteps=100, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ConditionalUNet1D(
            input_length=input_length,
            cond_dim=cond_dim,
            num_channels=4
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.diffusion = ConditionalDiffusionProcess(timesteps=timesteps, device=self.device)
        self.timesteps = timesteps

    @torch.no_grad()
    def generate(self, coord: torch.Tensor,x_t, batch_size=1):
        """
        参数:
            coord: 条件输入坐标 [B, 3]
        返回:
            cir: 生成的 CIR 序列 [B, 20, 4]
        """
        self.x_t = x_t

        coord = coord.to(self.device)               # [B, 3]
        x_t = torch.randn(batch_size, 4, 20).to(self.device)  # 初始化为纯噪声 [B, 4, 20]

        for t in reversed(range(self.x_t)):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # 模型预测当前时间步的噪声
            predicted_noise = self.model(x_t, coord, t_tensor)

            alpha_bar_t = self.diffusion.alpha_bars[t]
            alpha_bar_t_prev = self.diffusion.alpha_bars[t - 1] if t > 0 else None

            # 预测 x_0
            x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(alpha_bar_t_prev) * x_0_pred + torch.sqrt(1 - alpha_bar_t_prev) * noise
            else:
                x_t = x_0_pred

        cir = x_t.permute(0, 2, 1)  # [B, 4, 20] → [B, 20, 4]
        return cir
