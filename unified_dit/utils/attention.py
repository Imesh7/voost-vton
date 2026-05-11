from torch import nn
import torch


class JointAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv_1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.proj_c = nn.Linear(dim, dim)

    def forward(self, x, c):
        B, N, C = x.shape
        qkv_1 = (
            self.qkv_1(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        qkv_2 = (
            self.qkv_2(c)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q1, k1, v1 = qkv_1[0], qkv_1[1], qkv_1[2]
        q2, k2, v2 = qkv_2[0], qkv_2[1], qkv_2[2]
        
        q = torch.cat((q1, q2), dim=2)
        k = torch.cat((k1, k2), dim=2)
        v = torch.cat((v1, v2), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        output = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x , c = (
            output[:, :C, :],
            output[:, C :, :]
        )
        
        x = self.proj(x)
        x = self.proj_drop(x)
        c = self.proj_c(c)
        return x, c


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        output = (attn @ v).transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output