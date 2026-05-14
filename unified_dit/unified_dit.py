from torch import nn
import torch

from unified_dit.solver import Solver
from unified_dit.mm_dit_block.mm_dit_block import MMDiTBlock
from unified_dit.single_dit_block.single_dit_block import SingleDITBlock


class UnifiedDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.time_embed = SinosidualTimeEmbedding(config.hidden_size)
        self.mm_dit_block = MMDiTBlock(config)
        self.single_dit_block = SingleDITBlock(config)
        self.flow = FlowMatching(config)
        self.solver = Solver(config)
        

    # While inference
    @torch.inference_mode()
    def sample(self, task_token, image_token, timesteps):
        # Procee input function needed to create
        time_emb = self.time_embed(timesteps)
        x = self.mm_dit_block(task_token, image_token, time_emb)
        x = self.single_dit_block(x, time_emb)
        
        x = self.solver.sample(x)
        return x
    
    # while Training
    def forward(
        self,
        task_token: torch.Tensor,
        image_token: torch.Tensor,
        timesteps: torch.Tensor,
        x_0: torch.Tensor,  # noise
        x_1: torch.Tensor,  # clean image
        t: int,
    ):
        # process input
        time_emb = self.time_embed(timesteps)
        x = self.mm_dit_block(task_token, image_token, time_emb)
        x = self.single_dit_block(x, time_emb)
        
        x_t = x_0 * (1 - t) + t * x
        x_t = torch.cat([x_t, t], dim=1)
        v_t = self.flow(x_t)
        
        u_t = x - x_0
        
        loss = torch.mean((v_t - u_t) ** 2)
        return loss


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


class FlowMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.flow = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.CELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x_t):
        v_pred = self.flow(x_t)
        return v_pred  # Placeholder for actual output
