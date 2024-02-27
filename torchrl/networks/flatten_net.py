import torch
from .depth_route_net import DepthRouteNet

class FlattenDepthRouteNet(DepthRouteNet):
    def forward(self, input, idx, gate_sample=None, explore=True, return_gate=False):
        em_obs = input[0][..., -self.em_input_shape:]
        base_obs = input[0][..., :-self.em_input_shape]
        action = input[1]
        out = torch.cat([base_obs, action, em_obs], dim = -1)
        return super().forward(out, idx, gate_sample, explore, return_gate)