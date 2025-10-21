import torch
import torch.nn as nn
import dill
import re
import nltk
import pymorphy3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")

def inf_preprocess(text):

    morph = pymorphy3.MorphAnalyzer()
    stop_words = set(stopwords.words("russian")) - {'не', 'ну', 'вот'}
    # заменяем спецтокены временными плейсхолдерами
    text = re.sub(r"<MASK>", " MASKTOKEN ", text)
    text = re.sub(r"<SEP>", " SEPTOKEN ", text)

    tokens = word_tokenize(text, language="russian")
    sep_tokens = {".", "?", "!", "..."}
    lemmas = []

    for token in tokens:
        # возвращаем спецтокены обратно
        if token == "MASKTOKEN":
            lemmas.append("<MASK>")
        elif token == "SEPTOKEN":
            lemmas.append("<SEP>")
        elif token.isalpha() and token not in stop_words:
            lemmas.append(morph.parse(token)[0].normal_form)

    return " ".join(lemmas)

# Функция инференса
def mlm_infer(model, input_text, vocab, device, top_k=5):
    model.eval()
    with torch.no_grad():
        text = inf_preprocess(input_text)
        tokens = text.split()
        ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
        x = torch.tensor(ids).unsqueeze(0).to(device)
        mask = (x != vocab["<PAD>"]).long()

        hiddens = model(x, mask, return_hidden=True)
        logits = model.fc_mlm(hiddens)
        probs = torch.softmax(logits, dim=-1)

        inv_vocab = {v: k for k, v in vocab.items()}
        restored = []
        candidates = []
        for i, token in enumerate(tokens):
            if token == "<MASK>":
                top_preds = torch.topk(probs[0, i], k=top_k)
                candidates = [inv_vocab[idx.item()] for idx in top_preds.indices if idx != vocab['<UNK>']]
                restored.append(f"[{', '.join(candidates)}]")
            elif token == "<SEP>":  
                restored.append('.')
            else:
                restored.append(token)
        preprocessed_answer = " ".join(restored)
        return re.sub(r"<MASK>", str(candidates), input_text)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed, hidden, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed)
        self.norm2 = nn.LayerNorm(embed)
        self.ff = nn.Sequential(
            nn.Linear(embed, hidden),
            nn.GELU(),                 # GELU вместо ReLU
            nn.Dropout(0.1),
            nn.Linear(hidden, embed)
        )
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # MultiheadAttention принимает key_padding_mask (True = игнорировать токен!)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=~mask.bool() if mask is not None else None)
        x = self.norm1(x + self.dropout1(attn_out))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x

class Transformer_Max(nn.Module):
    def __init__(self, vocab_size, embed=128, num_classes=17, hidden=1024, num_heads=2, num_layers=2, max_len=500):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, embed)
        self.poses = nn.Embedding(max_len, embed)
        self.transes = nn.ModuleList([
            TransformerBlock(embed, hidden, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed, num_classes)

    def forward(self, x, mask=None):
        batch, seq_len = x.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device = x.device).unsqueeze(0).expand_as(x)
        x = self.tokens(x) + self.poses(pos)

        # маскируем паддинги
        mask_with_cls = None if mask is None else mask

        # Пропускаем через блоки
        for trans in self.transes:
            x = trans(x, mask=None if mask is None else mask_with_cls)

        x, _ = x.max(dim=1)
        return self.fc(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed=128, num_classes=17, hidden=1024, num_heads=2, num_layers=2, max_len=500):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size+1, embed)
        self.poses = nn.Embedding(max_len+1, embed)
        self.transes = nn.ModuleList([
            TransformerBlock(embed, hidden, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed, num_classes)
        self.cls = vocab_size

    def forward(self, x, mask=None):
        batch, seq_len = x.shape
        clses = torch.full((batch, 1), self.cls, dtype=torch.long, device=x.device)
        x = torch.cat((clses, x), dim=1)
        pos = torch.arange(0, seq_len+1, dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)
        x = self.tokens(x) + self.poses(pos)

        # маскируем паддинги
        mask_with_cls = None if mask is None else torch.cat([torch.ones((batch,1), device=x.device), mask], dim=1)

        # Пропускаем через блоки
        for trans in self.transes:
            x = trans(x, mask=None if mask is None else mask_with_cls)

        CLS = x[:, 0, :]
        return self.fc(CLS)

import dill

class Transformer_mix(nn.Module):
    def __init__(self, embed=1024, num_classes=17, hidden=2048, num_heads=16, num_layers=8, max_len=500, path_to_vocab=None, q99=120, classes=None, max_segments=100):
        super().__init__()
        self.vocab = dill.load(open(path_to_vocab, 'rb'))
        self.vocab_size = len(self.vocab)
        self.tokens = nn.Embedding(self.vocab_size+1, embed, padding_idx=0)
        self.poses = nn.Embedding(max_len+1, embed)
        self.segments = nn.Embedding(max_segments, embed)
        self.embed_dropout = nn.Dropout(0.1)
        self.transes = nn.ModuleList(
            TransformerBlock(embed, hidden, num_heads) for _ in range(num_layers)
        )
        self.layernorm = nn.LayerNorm(embed)
        self.fc = nn.Sequential(
            nn.Linear(embed*3, embed*6),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed*6, num_classes)
        )
        self.cls = self.vocab_size
        self.q99 = q99
        self.sep_id = self.vocab['<SEP>']
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
        #print(self.classes[pred])
        return pred

    def forward(self, x, mask=None, return_hidden=False):
        batch, seq = x.shape
        clses = torch.full((batch, 1), self.cls, dtype=torch.long, device=x.device)
        x = torch.cat([clses, x], dim=1)

        pos = torch.arange(0, seq+1, device=x.device).unsqueeze(0).expand(batch, -1)
        # сегменты (находим <SEP>, делаем cumsum)
        #sep_mask = (x == self.sep_id).int()       # где стоят <SEP>, будет 1
        #segments = sep_mask.cumsum(dim=1) % 2 # нумерация сегментов (0,1,2,...) кумулятивной суммой, то есть при каждом попадании единицы будет +1 к сумме, далее значения будут на 1 больше
        #segments = segments.clamp(max=self.segments.num_embeddings - 1) #модель рассчитана на 10 предложений, такой передан параметр, поэтому все предложения после 9 будут иметь индекс 9

        # складываем эмбеддинги
        #x = self.embed_dropout(self.tokens(x) + self.poses(pos) + self.segments(segments))
        x = self.embed_dropout(self.tokens(x) + self.poses(pos))
        # маскируем паддинги
        mask_with_cls = None if mask is None else torch.cat([torch.ones((batch,1), device=x.device), mask], dim=1)

        # Пропускаем через блоки
        for trans in self.transes:
            x = trans(x, mask=None if mask is None else mask_with_cls)
            
        x = self.layernorm(x)

        if return_hidden:
            return x[:, 1:, :]
        
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

class Classificator(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.ff(x)
        return x

from sklearn.metrics import f1_score, accuracy_score

def train(model, model_name:str, optimizer, loss_fn, train_loader, test_loader, scheduler, epochs, device):
    model, loss_fn = model.to(device), loss_fn.to(device)
    losses, f1s, accs = [], [], []
    best_acc, idx, best_idx, grad_norm = 0, 0, 0, 0
    for epoch in range(epochs):
        model.train()
        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device, dtype=torch.long)
            pred = model(x, mask)

            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            # ограничиваем градиенты перед шагом оптимизатора
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        f1, acc, val_loss = validate(model, loss_fn, test_loader, device)
        model.train()
        scheduler.step(val_loss)
        
        f1s.append(f1)
        accs.append(acc)
        losses.append(val_loss)
        idx += 1
        if acc>best_acc:
          torch.save(model.state_dict(), f'weights_{model_name}.pt')
          best_acc = acc
          best_idx = idx
          print(f"✅ Saved new best model to {f'weights_{model_name}.pt'}")
        print(f"Epoch {epoch+1}/{epochs} | loss={val_loss:.4f}, f1={f1:.4f}, acc={acc:.4f}")
    print(f'best_epoch {best_idx} with acc {best_acc}')
    return losses, f1s, accs


def validate(model, loss_fn, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    loss_sum, total = 0, 0

    with torch.no_grad():
        for x, mask, y in test_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device, dtype=torch.long)
            pred = model(x, mask)

            loss = loss_fn(pred, y)
            loss_sum += loss.item()

            preds = pred.argmax(dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            total += y.size(0)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return f1, acc, loss_sum / len(test_loader)
