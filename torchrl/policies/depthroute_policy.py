import torch
import torchrl.networks as networks
from .distribution import TanhNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class DepthRoutePolicy(networks.DepthRouteNet):
    def forward(self, x, idx=None, gate_sample=None, explore=True, return_gate=False):
        if return_gate:
            x, gates, gates_onehot, gates_softmax = super().forward(x, idx=idx, gate_sample=gate_sample, explore=explore, return_gate=return_gate)
        else:
            x = super().forward(x, idx=idx, gate_sample=gate_sample, explore=explore, return_gate=return_gate)

        mean, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        if return_gate:
            return mean, std, log_std, gates, gates_onehot, gates_softmax
        return mean, std, log_std

    def eval_act(self, x, idx=None, return_gate=True):
        with torch.no_grad():
            if return_gate:
                mean, std, log_std, gates, gates_onehot, gates_softmax = self.forward(x, idx, explore=False, return_gate=return_gate)
                dic = {
                    "gates": gates,
                    "gates_onehot": gates_onehot,
                    "gates_softmax": gates_softmax,
                }
            else:
                mean, std, log_std = self.forward(x, idx, explore=False)
                dic = {}

        action = torch.tanh(mean.squeeze(0))
        dic.update({
            "action": action
        })
        
        return dic

    def explore(self, x, idx=None, gate_sample=None, gate_explore=True, return_gate=True,
                    return_log_probs = False, return_pre_tanh = False):
                    
        if return_gate:
            mean, std, log_std, gates, gates_onehot, gates_softmax = self.forward(x, idx, gate_sample=gate_sample, explore=gate_explore, return_gate=return_gate)
            dic = {
                "gates": gates,
                "gates_onehot": gates_onehot,
                "gates_softmax": gates_softmax
            }
        else:
            mean, std, log_std = self.forward(x, idx, gate_sample=gate_sample, explore=gate_explore)
            dic = {}

        dis = TanhNormal(mean, std)

        ent = dis.entropy().sum(-1, keepdim=True) 
        
        dic.update({
            "mean": mean,
            "log_std": log_std,
            "ent":ent
        })

        if return_log_probs:
            action, z = dis.rsample( return_pretanh_value = True )
            log_prob = dis.log_prob(
                action,
                pre_tanh_value=z
            )
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            dic["pre_tanh"] = z.squeeze(0)
            dic["log_prob"] = log_prob
        else:
            if return_pre_tanh:
                action, z = dis.rsample( return_pretanh_value = True )
                dic["pre_tanh"] = z.squeeze(0)
            action = dis.rsample( return_pretanh_value = False )

        dic["action"] = action.squeeze(0)
        return dic