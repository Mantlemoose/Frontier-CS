import torch
import torch.nn.functional as F

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Baseline cross entropy implementation using PyTorch.
    
    Args:
        logits: Input tensor of shape (M, N) - logits for M samples and N classes
        targets: Input tensor of shape (M,) - target class indices
    
    Returns:
        Output tensor of shape (M,) - negative log-likelihood loss for each sample
    """
    return F.cross_entropy(logits, targets, reduction='none')

