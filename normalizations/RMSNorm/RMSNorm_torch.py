import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps: float = 1e-5):
        """Слой RMSNorm 

        Args:
            d_model (int): Размерность скрытого слоя
            eps (float, optional): смещение. Defaults to 1e-5.
        """
        
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones((d_model,)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход слоя

        Args:
            x (torch.Tensor): Входной батч

        Returns:
            torch.Tensor: Нормализованный батч
        """
        
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        out = x / rms * self.gamma
        return out
    

if __name__ == "__main__":
    B, S, H = 2, 10, 8
    
    x = torch.randn((B, S, H))*100
    
    my_rn = RMSNorm(H)
    torch_rn = nn.RMSNorm(H)
    
    assert torch.allclose(my_rn(x), torch_rn(x)), "Неправильная реализация"
    print("All OK")