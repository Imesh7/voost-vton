from torch import nn
import torch


class Solver:
    def __init__(self, flow: nn.Sequential, config):
        super().__init__()
        self.config = config
        self.flow = flow

    def sample(
        self,
        x: torch.Tensor,
        num_steps : int = 10,
    ):
        timesteps = torch.linspace(0, 1, steps=num_steps, device=x.device)

        for step in range(num_steps):
            time_cur = timesteps[step]
            time_next = timesteps[step + 1]

            v_pred = self.flow(x)
            x_0 = x - time_cur * v_pred
            x_1 = x + (1 - time_cur) * v_pred

            if step < num_steps - 1:
                x = (1.0 - time_next) * x_0 + time_next * x_1
            else:
                x = x_1

        return x
