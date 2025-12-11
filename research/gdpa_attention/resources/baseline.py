import torch
import math

def gdpa_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, GQ: torch.Tensor, GK: torch.Tensor) -> torch.Tensor:
    """
    Baseline GDPA attention implementation using PyTorch.
    
    Args:
        Q: Input tensor of shape (Z, H, M, Dq) - query tensor
        K: Input tensor of shape (Z, H, N, Dq) - key tensor
        V: Input tensor of shape (Z, H, N, Dv) - value tensor
        GQ: Input tensor of shape (Z, H, M, Dq) - query gate tensor
        GK: Input tensor of shape (Z, H, N, Dq) - key gate tensor
    
    Returns:
        Output tensor of shape (Z, H, M, Dv) - attention output
    """
    scale = 1.0 / math.sqrt(Q.shape[-1])
    Qg = Q * torch.sigmoid(GQ)
    Kg = K * torch.sigmoid(GK)
    scores = torch.matmul(Qg, Kg.transpose(-1, -2)) * scale
    P = torch.softmax(scores, dim=-1)
    O = torch.matmul(P, V).to(torch.float16)
    return O

