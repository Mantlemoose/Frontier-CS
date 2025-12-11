import torch

def bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Baseline batched matrix multiplication using PyTorch.
    
    Args:
        A: Input tensor of shape (B, M, K)
        B: Input tensor of shape (B, K, N)
    
    Returns:
        Output tensor of shape (B, M, N)
    """
    # A:[B,M,K], B:[B,K,N] -> [B,M,N]
    return torch.bmm(A.float(), B.float()).to(torch.float16)


