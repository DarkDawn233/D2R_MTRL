import torch.nn as nn

def get_parameters_num(param_list):
    return str(sum(p.numel() for p in param_list) / 1000) + 'K'


