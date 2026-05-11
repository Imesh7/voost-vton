import torch
import torch.nn as nn

from unified_dit.mm_dit_block.attention import Attention, JointAttention


class MMDiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layer_norm_task = nn.LayerNorm(config.hidden_size)
        self.layer_norm_image = nn.LayerNorm(config.hidden_size)

        # Scale and shift for task and image tokens
        self.scale_and_shift_task1 = AdaLN_Zero(config.hidden_size)
        self.scale_and_shift_imag1 = AdaLN_Zero(config.hidden_size)

        self.attention = JointAttention(
            config.hidden_size, num_heads=config.num_attention_heads
        )

        self.scale_and_shift_task2 = AdaLN_Zero(config.hidden_size)
        self.scale_and_shift_imag2 = AdaLN_Zero(config.hidden_size)
        
        self.mlp_task = MLP(config.hidden_size, mlp_ratio=config.mlp_ratio)
        self.mlp_image = MLP(config.hidden_size, mlp_ratio=config.mlp_ratio)
        
        self.scale_and_shift_task3 = AdaLN_Zero(config.hidden_size)
        self.scale_and_shift_imag3 = AdaLN_Zero(config.hidden_size)

    def forward(
        self,
        task_token: torch.Tensor,
        image_token: torch.Tensor,
        time_emb: torch.Tensor,
    ):
        task_token_norm = self.layer_norm_task(task_token)
        image_token_norm = self.layer_norm_image(image_token)

        # shift & scale 1
        task_mod, task_gate = self.scale_and_shift_task1(task_token_norm, time_emb)
        image_mod, image_gate = self.scale_and_shift_imag1(image_token_norm, time_emb)

        # Apply gate & residual connection
        task_mod = task_mod * (1 + task_gate.unsqueeze(1)) + task_token
        image_mod = image_mod * (1 + image_gate.unsqueeze(1)) + image_token

        output = self.attention(task_mod, image_mod)

        # shift & scale 2
        task_mod, task_gate = self.scale_and_shift_task2(task_token_norm, time_emb)
        image_mod, image_gate = self.scale_and_shift_imag2(image_token_norm, time_emb)

        # Apply gate & residual connection
        task_mod = task_mod * (1 + task_gate.unsqueeze(1)) + task_token
        image_mod = image_mod * (1 + image_gate.unsqueeze(1)) + image_token
        
        task_mlp = self.mlp_task(task_mod)
        image_mlp = self.mlp_image(image_mod)

        # shift & scale 3
        task_mod, task_gate = self.scale_and_shift_task3(task_mlp, time_emb)
        image_mod, image_gate = self.scale_and_shift_imag3(image_mlp, time_emb)

        # Apply gate & residual connection
        task_mod = task_mod * (1 + task_gate.unsqueeze(1)) + task_token
        image_mod = image_mod * (1 + image_gate.unsqueeze(1)) + image_token
        
        return task_token, image_token  # Placeholder for actual output


class AdaLN_Zero(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale_and_shift = nn.Linear(hidden_size, hidden_size * 3)

    def forward(self, x1, time_emb):
        t_mod = self.scale_and_shift(time_emb)
        x1_scale, x1_shift, x1_gate = t_mod.chunk(3, dim=-1)
        return (
            x1 * (1 + x1_scale.unsqueeze(1)) + x1_shift.unsqueeze(1)
        ), x1_gate  # 2 outputs: modulated x1 and gate for x1



class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, int(hidden_size * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(hidden_size * mlp_ratio), hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x