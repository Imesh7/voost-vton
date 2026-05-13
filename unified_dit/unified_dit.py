from torch import nn
import torch

from unified_dit.mm_dit_block.mm_dit_block import MMDiTBlock
from unified_dit.single_dit_block.single_dit_block import SingleDITBlock


class UnifiedDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.time_embed = SinosidualTimeEmbedding(config.hidden_size)
        self.mm_dit_block = MMDiTBlock(config)
        self.single_dit_block = SingleDITBlock(config)

    def forward(
        self,
        task_token: torch.Tensor,
        image_token: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        time_emb = self.time_embed(timesteps)
        x = self.mm_dit_block(task_token, image_token, time_emb)
        x = self.single_dit_block(x, time_emb)
        return x  # Placeholder for actual output


class SinosidualTimeEmbedding(nn.Module):
    def __init__(self, hidden_size, max_period=10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_period = max_period

    def forward(self, timesteps):
        half_dim = self.hidden_size // 2
        emb = torch.log(torch.tensor(self.max_period)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb
