import torch
import math

def ragged_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, row_lens: torch.Tensor) -> torch.Tensor:
    """
    PyTorch baseline for ragged attention.
    
    Q:[M,D], K:[N,D], V:[N,Dv], row_lens:[M] -> O:[M,Dv]
    """
    M, D = Q.shape
    N = K.shape[0]
    scale = 1.0 / math.sqrt(D)

    idx = torch.arange(N, device=Q.device)
    mask = idx.unsqueeze(0) < row_lens.to(idx.dtype).unsqueeze(1)

    scores = (Q @ K.T) * scale          # [M,N]
    scores = scores.masked_fill(~mask, float("-inf"))
    P = torch.softmax(scores, dim=-1)
    O = (P @ V).to(torch.float16)
    return O

