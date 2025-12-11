import torch
import torch.nn.functional as F

def fused_linear_jsd(X: torch.Tensor, W1: torch.Tensor, B1: torch.Tensor, W2: torch.Tensor, B2: torch.Tensor) -> torch.Tensor:
    """
    Baseline fused linear Jensen-Shannon Divergence implementation using PyTorch.
    
    Args:
        X: Input tensor of shape (M, K) - input features (float16)
        W1: Weight tensor of shape (K, N) - first weight matrix (float16)
        B1: Bias tensor of shape (N,) - first bias vector (float32)
        W2: Weight tensor of shape (K, N) - second weight matrix (float16)
        B2: Bias tensor of shape (N,) - second bias vector (float32)
    
    Returns:
        Output tensor of shape (M,) - Jensen-Shannon Divergence per sample (float32)
    """
    logits1 = (X.float() @ W1.float()) + B1.float()
    logits2 = (X.float() @ W2.float()) + B2.float()
    P = torch.softmax(logits1, dim=-1)
    Q = torch.softmax(logits2, dim=-1)
    Mmid = 0.5 * (P + Q)
    eps = 1e-12
    jsd = 0.5 * (torch.sum(P * (torch.log(P + eps) - torch.log(Mmid + eps)), dim=-1) +
                 torch.sum(Q * (torch.log(Q + eps) - torch.log(Mmid + eps)), dim=-1))
    return jsd

