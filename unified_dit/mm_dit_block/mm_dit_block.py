import torch
import torch.nn as nn

from unified_dit.utils.attention import Attention, JointAttention
from unified_dit.utils.mlp import MLP
from unified_dit.utils.rope import RoPE


class MMDiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layer_norm_task = nn.LayerNorm(config.hidden_size)
        self.layer_norm_image = nn.LayerNorm(config.hidden_size)

        # Scale and shift for task and image tokens
        self.scale_and_shift_task1 = AdaLN_Zero(config.hidden_size)
        self.scale_and_shift_img1 = AdaLN_Zero(config.hidden_size)

        self.joint_attention = JointAttention(
            config.hidden_size, num_heads=config.num_attention_heads
        )

        self.scale_and_shift_task2 = AdaLN_Zero(config.hidden_size)
        self.scale_and_shift_img2 = AdaLN_Zero(config.hidden_size)

        self.mlp_task = MLP(config.hidden_size, mlp_ratio=config.mlp_ratio)
        self.mlp_image = MLP(config.hidden_size, mlp_ratio=config.mlp_ratio)

        self.scale_and_shift_task3 = AdaLN_Zero(config.hidden_size)
        self.scale_and_shift_img3 = AdaLN_Zero(config.hidden_size)

        self.rope = RoPE(config.hidden_size // config.num_attention_heads)

    def forward(
        self,
        task_token: torch.Tensor,
        image_token: torch.Tensor,
        time_emb: torch.Tensor,
    ):
        # Apply RoPE to task_token
        pos_emb_task = self.rope.get_pos_emb(task_token) # this returns (sin_emb, cos_emb)
        x1 = self.rope.apply_rotary_pos_emb(task_token, pos_emb_task)
        
        pos_emb_image = self.rope.get_pos_emb(image_token) # this returns (sin_emb, cos_emb)
        x2 = self.rope.apply_rotary_pos_emb(image_token, pos_emb_image)
        
        task_token_norm = self.layer_norm_task(x1)
        image_token_norm = self.layer_norm_image(x2)

        # shift & scale 1
        task_mod, task_gate = self.scale_and_shift_task1(task_token_norm, time_emb)
        image_mod, image_gate = self.scale_and_shift_img1(image_token_norm, time_emb)

        # Apply gate & residual connection
        task_mod = task_mod * (1 + task_gate.unsqueeze(1)) + task_token
        image_mod = image_mod * (1 + image_gate.unsqueeze(1)) + image_token

        output_task, output_image = self.joint_attention(task_mod, image_mod)

        # shift & scale 2
        task_mod, task_gate = self.scale_and_shift_task2(output_task, time_emb)
        image_mod, image_gate = self.scale_and_shift_img2(output_image, time_emb)

        # Apply gate & residual connection
        task_mod = task_mod * (1 + task_gate.unsqueeze(1)) + task_token
        image_mod = image_mod * (1 + image_gate.unsqueeze(1)) + image_token

        task_mlp = self.mlp_task(task_mod)
        image_mlp = self.mlp_image(image_mod)

        # shift & scale 3
        task_mod, task_gate = self.scale_and_shift_task3(task_mlp, time_emb)
        image_mod, image_gate = self.scale_and_shift_img3(image_mlp, time_emb)

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


