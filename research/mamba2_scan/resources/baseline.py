import torch

def chunk_scan(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, chunk: int = 128, BD: int = 128) -> torch.Tensor:
    """
    Baseline Mamba2 chunked scan implementation using PyTorch.
    
    Args:
        X: Input tensor of shape (L, D) - input sequence
        A: Input tensor of shape (L, D) - decay factors
        B: Input tensor of shape (L, D) - input weights
        chunk: Chunk size for parallel processing (default 128)
        BD: Block dimension for feature dimension tiling (default 128) - unused in baseline
    
    Returns:
        Output tensor of shape (L, D) - scan output
    """
    # y_t = a_t * y_{t-1} + b_t * x_t
    L, D = X.shape
    y = torch.zeros(D, device=X.device, dtype=torch.float32)
    out = torch.empty(L, D, device=X.device, dtype=torch.float32)
    for t in range(L):
        y = A[t].float() * y + B[t].float() * X[t].float()
        out[t] = y
    return out.to(torch.float16)


