from torch import nn

from unified_dit.mm_dit_block.mm_dit_block import MLP, AdaLN_Zero
from unified_dit.utils.attention import Attention
from unified_dit.utils.rope import RoPE


class SingleDITBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layer_norm_task = nn.LayerNorm(config.hidden_size)
        self.scale_and_shift_task1 = AdaLN_Zero(config.hidden_size)

        self.attention = Attention(
            config.hidden_size, num_heads=config.num_attention_heads
        )

        self.mlp = MLP(config.hidden_size, mlp_ratio=config.mlp_ratio)
        self.scale_and_shift_task2 = AdaLN_Zero(config.hidden_size)

        self.rope = RoPE(config.hidden_size // config.num_attention_heads)

    def forward(
        self,
        task_token,
        time_emb,
    ):
        # Apply RoPE to task_token
        pos_emb = self.rope.get_pos_emb(task_token) # this returns (sin_emb, cos_emb)
        x = self.rope.apply_rotary_pos_emb(task_token, *pos_emb)
        
        # Layer norm
        task_token_norm = self.layer_norm_task(x)

        # shift & scale 1
        task_mod, task_gate = self.scale_and_shift_task1(task_token_norm, time_emb)

        # Apply gate & residual connection
        task_mod = task_mod * (1 + task_gate.unsqueeze(1)) + task_token


        cos_pos_emb , sin_pos_emb = self.rope.get_pos_emb(task_mod)
        task_mod = self.rope.apply_rotary_pos_emb(task_mod, cos_pos_emb, sin_pos_emb, (sin_pos_emb, cos_pos_emb))

        # Attention
        output_task = self.attention(task_mod)

        # MLP
        mlp = self.mlp(output_task)

        # shift & scale 2
        task_mod, task_gate = self.scale_and_shift_task2(mlp, time_emb)

        # Apply gate & residual connection
        task_mod = task_mod * (1 + task_gate.unsqueeze(1)) + task_token
        return task_mod
