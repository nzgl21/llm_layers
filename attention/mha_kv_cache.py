import torch
from torch import nn
from torch.functional import F


class AttentionLayer(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_kv_cache: bool = True
    ):
        """Инициализация MHA слоя с возможностью KV кэша

        Args:
            d_model (int): Размерность скрытого слоя
            num_heads (int): Кол-во голов
            dropout (float, optional): Дропаут. Defaults to 0.1.
            use_kv_cache (bool, optional): Использоват ли kv cache на инференсе. Defaults to True.
        """
        
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_kv_cache = use_kv_cache
        self.d_k = d_model // num_heads
        
        assert self.d_k * num_heads == d_model
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        

    def _div_on_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Разделение на головы

        Args:
            x (torch.Tensor): Вход размера (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Выход размера (batch_size, num_heads, seq_len, d_model)
        """
        
        B, S, _ = x.shape
        
        # Разделяем последнюю ось, меняем оси.
        x = x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        
        return x
    
    
    def forward(
        self,
        batch: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Прямой проход. Для eval используется кэширование.

        Args:
            batch (torch.Tensor): Входной тензор
            past_kv (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Прошлый Кэш. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: Выход
        """     
        
        B, S, _ = batch.shape
        
        # Проекции. (B, S, D)
        q = self.W_q(batch)
        k = self.W_k(batch)
        v = self.W_v(batch)
        
        # (B, H, S, d_k)
        q = self._div_on_heads(q)
        k = self._div_on_heads(k)
        v = self._div_on_heads(v)
        
        if past_kv is not None:
            k_past, v_past = past_kv
            k = torch.cat([k_past, k], dim=-2)
            v = torch.cat([v_past, v], dim=-2)
        
        cur_kv = (k, v)
        total_seq_len = k.shape[-2]
        
        scores = q @ k.transpose(-1, -2) / self.d_k ** 0.5
        
        # Казуальная маска.
        mask = torch.triu(torch.full((S, total_seq_len), fill_value=float('-inf'), device=batch.device), diagonal=1)
        scores += mask
        
        probs = F.softmax(scores, dim=-1)
        att = probs @ v
        
        out = att.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.W_o(out)
        
        return out, cur_kv

        
        
        
        
        
        
        
        