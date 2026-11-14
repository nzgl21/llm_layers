import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5):
        """
        Слой Layer Normalization.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((d_model,)))
        self.bias = nn.Parameter(torch.zeros((d_model,)))
        self.eps = eps
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Прямой проход Layer Normalization.
        Args:
            batch (torch.Tensor): Входной тензор формы (B, S, H) / [batch_size, seq_len, d_model].
        Returns:
            torch.Tensor: Нормализованный тензор той же формы (B, S, H).
        """
        
        means = batch.mean(dim=-1, keepdim=True)
        var = batch.var(dim=-1, keepdim=True, unbiased=False)
        ans = (batch - means) / torch.sqrt(var + self.eps) * self.gamma + self.bias
        
        return ans



        