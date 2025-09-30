import torch
p = torch.Tensor([0.1, 0.2, 0.7])
q = torch.Tensor([0.333, 0.334, 0.333])

def kl_div(P, Q):
    return (P * (P / Q).log()).sum()

kl_div(p, q)