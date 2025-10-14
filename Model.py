import torch
import torch.nn as nn
import dill

class TransformerBlock(nn.Module):
    def __init__(self, embed, hidden, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed)
        self.norm2 = nn.LayerNorm(embed)
        self.ff = nn.Sequential(
            nn.Linear(embed, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed)
        )
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x, mask=None):
        # MultiheadAttention принимает key_padding_mask (True = игнорировать токен!)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=~mask.bool() if mask is not None else None)
        x = self.norm1(x + self.dropout1(attn_out))
        x = self.norm2(x + self.dropout1(self.ff(x)))
        return x

class Transformer_mix(nn.Module):
    def __init__(self, vocab_size=None, embed=512, num_classes=17, hidden=1024, num_heads=16, num_layers=3, max_len=500, path_to_vocab=None, q99=120, classes=None, max_segments=100):
        super().__init__()
        self.vocab = None if path_to_vocab is None else dill.load(open(path_to_vocab, "rb"))
        self.vocab_size = len(self.vocab) if vocab_size is None else vocab_size
        self.tokens = nn.Embedding(self.vocab_size+1, embed)
        self.poses = nn.Embedding(max_len+1, embed)
        #self.segments = nn.Embedding(max_segments, embed)
        self.transes = nn.ModuleList(
            TransformerBlock(embed, hidden, num_heads) for _ in range(num_layers)
        )
        self.fc = nn.Sequential(
            nn.Linear(embed*3, embed*6),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed*6, num_classes)
        )
        self.cls = self.vocab_size
        self.q99 = q99
        #self.sep_id = None if self.vocab is None else self.vocab.get('<SEP>', None)
        self.classes = classes

    def predict(self, text, device):
        # Индексы токенов
        indices_tr = [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.split()] 
        indices_tr_q99 = indices_tr[:self.q99]  # обрезка до q99
        # Паддинг
        pad_len = self.q99 - len(indices_tr_q99)
        indices_tr_q99 = indices_tr_q99 + [self.vocab["<PAD>"]] * pad_len
        # Маска: 1 = реальный токен, 0 = паддинг
        mask = [1] * (len(indices_tr[:self.q99])) + [0] * pad_len #indices_tr хранит предложения без падинга, так что по его длине максируем [1] и [0] падинги, но отсекаем по q99, если длинный попался
        # В тензоры
        x = torch.tensor(indices_tr_q99, dtype=torch.long).unsqueeze(0).to(device)  # (1, q99)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(device)       # (1, q99)
        # Предсказание
        with torch.no_grad():
            logits = self.forward(x, mask)
            pred = torch.argmax(logits, dim=1).item()
        return pred

    def forward(self, x, mask=None):
        batch, seq = x.shape
        clses = torch.full((batch, 1), self.cls, dtype=torch.long, device=x.device)
        x = torch.cat([clses, x], dim=1)

        pos = torch.arange(0, seq+1, device=x.device).unsqueeze(0).expand(batch, -1)
        # сегменты (находим <SEP>, делаем cumsum)
        #sep_mask = (x == self.sep_id).int()       # где стоят <SEP>, будет 1
        #segments = sep_mask.cumsum(dim=1) % 2 # нумерация сегментов (0,1,2,...) кумулятивной суммой, то есть при каждом попадании единицы будет +1 к сумме, далее значения будут на 1 больше
        #segments = segments.clamp(max=self.segments.num_embeddings - 1) #модель рассчитана на 10 предложений, такой передан параметр, поэтому все предложения после 9 будут иметь индекс 9

        # складываем эмбеддинги
        #x = self.tokens(x) + self.poses(pos) + self.segments(segments)
        x = self.tokens(x) + self.poses(pos)
        # маскируем паддинги
        mask_with_cls = None if mask is None else torch.cat([torch.ones((batch,1), device=x.device), mask], dim=1)

        # Пропускаем через блоки
        for trans in self.transes:
            x = trans(x, mask=None if mask is None else mask_with_cls)

        if mask is not None:
            x_masked = x[:,1:,:] * mask.unsqueeze(-1)   # обнуляем паддинги
            mean_pooling = x_masked.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            x_masked_for_max = x[:, 1:, :].masked_fill(mask.unsqueeze(-1) == 0, -1e9) #шумы, прошедшие через модель, по сути фейки, нельзя, чтобы выбрались как максимум вместо реальных данных
            max_pooling, _ = x_masked_for_max.max(1)
        else:
            mean_pooling = x[:,1:,:].mean(1)
            max_pooling, _ = x[:,1:,:].max(1)

        CLS = x[:,0,:]
        out = torch.cat([CLS, mean_pooling, max_pooling], dim=1)
        return self.fc(out)