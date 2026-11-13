import numpy as np

class MultiHeadAttentionLayer:

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float = 0.1,
            train = True
    ):
        """
        Инициализирует модуль Self-MHA

        Параметры
        ------------
        d_model : int
            Общая размерность входных эмбеддингов.
        num_heads : int
            Количество параллельных голов внимания.
        dropout : float, optional
            Вероятность дропаута (по умолчанию 0.1).
        """

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout =dropout
        self.d_k = d_model // num_heads
        self.train = train

        assert d_model == (self.d_k * num_heads), "d_model должен на цело делиться на num_heads"

        

        self.W_q = self._xavier_matrix_init()
        self.W_k = self._xavier_matrix_init()
        self.W_v = self._xavier_matrix_init()
        self.W_o = self._xavier_matrix_init()


    def forward(
            self,
            q: np.ndarray,
            k: np.ndarray,
            v: np.ndarray,
            mask = None
    ):
        """
        Прямой проход по MHA

        ---------------------
        Параметры
        q: Тензор запросов (batch_size, seq_len, d_model)
        k: Тензор ключей (batch_size, seq_len, d_model)
        v: Тензор значений (batch_size, seq_len, d_model)
        mask: Тензор маски (batch_size, 1, seq_len, seq_len) или (seq_len, seq_len)
        """

        B, S, _ = q.shape

        # Применение операторов
        Q = q @ self.W_q
        K = k @ self.W_k
        V = v @ self.W_v
        
        # Разделение на головы
        # (B, H, S, d_k)
        Q = self._div_by_heads(Q)
        K = self._div_by_heads(K)
        V = self._div_by_heads(V)
        
        scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(self.d_k)       # (B, H, S, S)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
            
        probs = self._softmax(scores)
        probs = self._dropout(probs)
        
        att = probs @ V                                             # (B, H, S, d_k)
        att = np.ascontiguousarray(att.swapaxes(1, 2)).reshape((B, S, self.d_model))
        return att @ self.W_o
        
        
        
    def __call__(self, *args, **kwargs):
        """
        Прямой проход по MHA

        ---------------------
        Параметры
        q: Тензор запросов (batch_size, seq_len, d_model)
        k: Тензор ключей (batch_size, seq_len, d_model)
        v: Тензор значений (batch_size, seq_len, d_model)
        mask: Тензор маски (batch_size, 1, seq_len, seq_len) или (seq_len, seq_len)
        """
        return self.forward(*args, **kwargs)
        

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
            Математически мтабильный Softmax: exp(s_ij - max(s_i1, s_i2,...))
        """

        max_i = x.max(axis=axis, keepdims=True)
        x = x - max_i
        exp_x = np.exp(x) 
        probs = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        return probs

    def _xavier_matrix_init(self) -> np.ndarray:
        """Инициализирует веса по схеме Xavier Normal: N(0, 2 / (fan_in + fan_out))"""
        
        std = 1.0 / np.sqrt(self.d_model)
        return np.random.normal(0.0, std, size=(self.d_model, self.d_model))

    def _div_by_heads(self, x: np.ndarray):
        """Делит Тензор для обработки несколькими головами"""

        B, S, _ = x.shape   
        return x.reshape((B, S, self.num_heads, self.d_k)).transpose(0, 2, 1, 3)

    def _dropout(self, x: np.ndarray) -> np.ndarray:
        """
        Применяет слой inverted dropout
        
        ----------
        Параметры
        x: Входной тензор
        """
        
        if not self.train or self.dropout == 0.0:
            return x
        
        keep_prob = 1 - self.dropout
        mask = (np.random.rand(*x.shape) < keep_prob)
        return (x * mask) / keep_prob