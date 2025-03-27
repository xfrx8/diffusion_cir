import torch
import torch.nn as nn

class ConditionalDiffusionProcess:
    def __init__(self, timesteps, device='cpu'):
        super().__init__()
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x_0, return_noise=False):
        """
        分步加噪。默认返回最后一个时间步的 x_t 和噪声 epsilon
        如果 return_noise=True，返回 x_t, noise，否则只返回 x_t
        """
        t = self.timesteps - 1
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bars[t]

        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

        if return_noise:
            return x_t, noise
        else:
            return x_t

    def get_x_t(self, x_0, t, noise):
        """
        训练用的：任意时间步 t 上加噪：x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        参数:
            x_0: 原始输入 [B, C, L]
            t: 时间步张量 [B] 或 int
            noise: 随机噪声 [B, C, L]
        返回:
            x_t: 添加噪声后的样本
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.long, device=x_0.device)

        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        # [B] -> [B, 1, 1] 用于广播
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1).to(x_0.device)

        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t

    def get_x_t(self, x_0, t, noise):
        """
        训练用的：任意时间步 t 上加噪：x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        参数:
            x_0: 原始输入 [B, C, L]
            t: 时间步张量 [B] 或 int
            noise: 随机噪声 [B, C, L]
        返回:
            x_t: 添加噪声后的样本
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.long, device=x_0.device)

        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        # [B] -> [B, 1, 1] 用于广播
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1).to(x_0.device)

        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t

    def reverse_diffusion(self, model, x_T, cond=None):
        """
        分步去噪，使用模型预测每一步的噪声
        """
        x_t = x_T

        for t in reversed(range(self.timesteps)):
            # 构造当前时间步 t 的张量表示
            t_tensor = torch.full((x_t.size(0),), t, dtype=torch.long, device=x_t.device)
            predicted_noise = model(x_t, cond, t_tensor)

            # 当前时间步的累积 α_bar 参数
            alpha_bar_t = self.alpha_bars[t]

            # 使用累积参数预测 x₀
            x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

            # 计算 x_{t-1}，若 t > 0 则使用 t-1 时刻的累积 α_bar
            if t > 0:
                noise = torch.randn_like(x_t)
                alpha_bar_prev = self.alpha_bars[t - 1]
                x_t = (
                        torch.sqrt(alpha_bar_prev) * x_0_pred +
                        torch.sqrt(1 - alpha_bar_prev) * noise
                )
            else:
                x_t = x_0_pred  # 当 t == 0 时直接返回预测的 x₀

        return x_t