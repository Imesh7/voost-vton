from torch import nn
import torch

from unified_dit.mm_dit_block.mm_dit_block import MMDiTBlock
from unified_dit.single_dit_block.single_dit_block import SingleDITBlock


class UnifiedDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mm_dit_block = MMDiTBlock(config)
        self.single_dit_block = SingleDITBlock(config)

    def forward(self, task_token: torch.Tensor, image_token: torch.Tensor, time_emb: torch.Tensor):
        x = self.mm_dit_block(task_token, image_token, time_emb)
        x = self.single_dit_block(x, time_emb)
        return x  # Placeholder for actual output