import torch
import math

def flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """
    Baseline flash attention implementation using PyTorch.
    
    Args:
        Q: Input tensor of shape (Z, H, M, Dq) - query tensor
        K: Input tensor of shape (Z, H, N, Dq) - key tensor
        V: Input tensor of shape (Z, H, N, Dv) - value tensor
        causal: Whether to apply causal masking (default True)
    
    Returns:
        Output tensor of shape (Z, H, M, Dv) - attention output
    """
    # Q:[Z,H,M,D], K:[Z,H,N,D], V:[Z,H,N,Dv]
    Z, H, M, D = Q.shape
    N = K.shape[-2]
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale  # [Z,H,M,N]
    if causal:
        mask = torch.ones((M, N), device=Q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    P = torch.softmax(scores, dim=-1)
    O = torch.matmul(P, V).to(torch.float16)
    return O

