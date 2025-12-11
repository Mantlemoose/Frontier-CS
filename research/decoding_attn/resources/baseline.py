import torch
import math

def decoding_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Baseline decoding attention implementation using PyTorch.
    
    Args:
        Q: Input tensor of shape (Z, H, M, Dq) - query tensor
        K: Input tensor of shape (Z, H, N, Dq) - key tensor
        V: Input tensor of shape (Z, H, N, Dv) - value tensor
    
    Returns:
        Output tensor of shape (Z, H, M, Dv) - attention output
    """
    # Q:[Z,H,M,D], K:[Z,H,N,D], V:[Z,H,N,Dv]
    Z, H, M, D = Q.shape
    N = K.shape[-2]
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale  # [Z,H,M,N]
    P = torch.softmax(scores, dim=-1)
    O = torch.matmul(P, V).to(torch.float16)
    return O

