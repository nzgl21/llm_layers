import torch
import torch.nn.functional as F


def top_p_sampling(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0
) -> torch.Tensor:
    """Top-p семплирование

    Args:
        logits (torch.Tensor): Тензор логитов размера (vocab_size, ) или (batch_size, vocab_size)
        p (float, optional): Параметр семплиорвания. Defaults to 0.9.
        temperature (float, optional): Температура семплирования. Defaults to 1.0.

    Returns:
        torch.Tensor: Тензор выбранных Token ID.
    """
    
    assert 0.0 < p <= 1.0
    assert temperature > 0.0
    
    if temperature != 1.0:
        logits = logits / temperature
    
    #Сортируем логиты, получаем вероятности
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    #Находим префиксные суммы, чтобы отсеять лишние токены
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    #Составляем маску отсечения. При этом токен, при котором cumsum
    #превысил p надо тоже включить. Первый токен так же включаем в любом случае.
    mask = cumsum_probs > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    
    # Заново пересчитываем вероятности с учетом маски
    masked_logits = sorted_logits.masked_fill(mask, value=float('-inf'))
    masked_probs = F.softmax(masked_logits, dim=-1)
    
    #Выбираем токен, получаем его idx в отсортированном Тензоре
    sampling = torch.multinomial(masked_probs, num_samples=1)
    
    # Находим индекс в исходном словаре
    out = torch.gather(sorted_idx, dim=-1, index=sampling)
    return out



# Мини тестирование
if __name__ == "__main__":
    x = torch.tensor(
        [
            [1, 1, 2, 2, 10, 0],        
            [10, 1, 1, 1, 1, 1],        
            [1, 1, 1, 1, 1, 10]
        ],
        dtype=torch.float32
    )
    
    print(
        top_p_sampling(
            x,
            0.1,
            1
        )
    )
    