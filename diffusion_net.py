import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 可学习的位置编码 ---
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, length, channels):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, channels, length))

    def forward(self, x):
        return x + self.pe

# --- Sinusoidal 时间嵌入 ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

# --- 通道注意力模块 ---
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, L]
        y = self.avg_pool(x).squeeze(-1)  # [B, C]
        y = self.fc(y).unsqueeze(-1)     # [B, C, 1]
        return x * y

# --- 改进后的 Conditional UNet1D ---
class ConditionalUNet1D(nn.Module):
    def __init__(self, input_length, cond_dim, num_channels=2, time_embed_dim=64):
        super().__init__()
        self.input_length = input_length
        self.num_channels = num_channels
        self.skip_proj = nn.Conv1d(num_channels, 8, kernel_size=1)
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1))

        # Learnable Positional Encoding
        self.pos_enc = LearnablePositionalEncoding(input_length, num_channels)

        # 条件向量投影
        self.cond_embed = nn.Linear(cond_dim, num_channels * input_length)

        # 时间嵌入模块
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU()
        )
        self.time_to_input = nn.Linear(64, input_length)
        self.time_to_middle = nn.Linear(64, 16)

        # --- 下采样层：Conv → Pool ---
        self.down1 = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(2)
        )

        # --- 多尺度 Dilated 中间层 ---
        self.middle = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )

        # 注意力模块
        self.attn = ChannelAttention(16)

        # --- 上采样层 ---
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=2, stride=2),
            nn.BatchNorm1d(8),
            nn.GELU()
        )

        # 最终输出映射
        self.final = nn.Conv1d(8, num_channels, kernel_size=1)

    def forward(self, x, cond, t):
        B = x.size(0)

        # 条件嵌入
        cond_emb = self.cond_embed(cond).view(B, self.num_channels, self.input_length)

        # 时间步嵌入
        t_emb = self.time_embedding(t)  # [B, 64]
        t_input = self.time_to_input(t_emb).unsqueeze(1).expand(-1, self.num_channels, -1)
        t_middle = self.time_to_middle(t_emb).unsqueeze(-1)

        # 输入：原始信号 × scale + 位置 + 条件 + 时间
        x_scaled = x * self.scale
        x_pos = self.pos_enc(x_scaled)
        x_input = x_pos + cond_emb + t_input
        res = x_input  # 保留 skip/residual 用

        # 下采样
        d1 = self.down1(x_input)

        # 中间层 + 时间调制 + 注意力
        m = self.middle(d1) + t_middle.expand(-1, 16, d1.shape[-1])
        m = self.attn(m)

        # 上采样
        u1 = self.up1(m)

        # 尺寸对齐（插值）+ 残差
        if u1.shape[-1] != res.shape[-1]:
            u1 = F.interpolate(u1, size=res.shape[-1])
        res_proj = self.skip_proj(res)  # [B, 8, L]
        out = self.final(u1 + res_proj)

        return out
