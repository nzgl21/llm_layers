from torch.functional import F
from torch import nn
import torch


def min_p_sampling(
    logits: torch.Tensor,
    min_p: float
) -> torch.Tensor:
    """Min p семплирование
        Токены семплируются при условии что p_i > min_p * p_max
    Args:
        logits (torch.Tensor): Тензор логитов модели размера (vocab_size,) или (batch_size, vocab_size)
        min_p (float): Параметр семплирования.

    Returns:
        torch.Tensor: _description_
    """
    
    assert 0.0 < min_p
    
    # Считаем вероятности
    probs = F.softmax(logits, dim=-1)
    
    # Порог отсечения
    threshold = torch.max(probs, dim=-1, keepdim=True)[0] * min_p
    #Зануляем вероятности
    probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
    # Перенормируем вероятности
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Семплиируем
    out = torch.multinomial(probs, num_samples=1)
    return out


# if __name__ == "__main__":
    
#     x = torch.randn((8,))
    
#     print(x)
    
#     print(min_p_sampling(x, 0.5))