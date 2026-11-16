import torch
from torch import nn
from torch.functional import F


def top_k_sampling(
    logits: torch.Tensor,
    k: int = 1,
    temperature: float = 1.0
) -> torch.Tensor:
    """Топ k семплирование с температурой

    Args:
        logits (torch.Tensor): Тензор логитов размерности (vocab_size, ) или (batch_size, vocab_size)
        k (int, optional): Параметр семплирования. Defaults to 5.
        temperature (float, optional): Температура. Defaults to 1.0.

    Returns:
        torch.Tensor: Тензор из семплированных token ID
    """
    
    vocab_size = logits.shape[-1]
    
    assert 0 < k <= vocab_size, f"k должен быть в пределах (0, {vocab_size}]"
    assert temperature > 0, "Температура должна быть положительной!"
    
    #Оставляем top k элементов
    sorted_logits, sorted_idx = logits.topk(k=k, dim=-1)
    
    if temperature != 1.0:
        logits = logits / temperature
    
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    sampled = sorted_probs.multinomial(num_samples=1)
    
    out = torch.gather(sorted_idx, dim=-1, index=sampled)
    return out



if __name__ == "__main__":
    
    x = torch.randn((2, 8))    
    print(top_k_sampling(x, temperature=1))
    