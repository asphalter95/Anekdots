# Anekdots

## 1. Описание решения
Модели Transformer для классификации анекдотов по 17 классам
0: 'aforizmi',
1: 'meditsinskie',
2: 'narodnie',
3: 'poshlie-i-intimnie',
4: 'pro-alkogolikov',
5: 'pro-armiu',
6: 'pro-detey',
7: 'pro-evreev',
8: 'pro-militsiyu',
9: 'pro-mugchin',
10: 'pro-novih-russkih',
11: 'pro-semyu',
12: 'pro-studentov',
13: 'pro-vovochku',
14: 'raznie',
15: 'shkolnie-i-pro-shkolu',
16: 'tsitati'

Модели обучены на датасете русских анекдотов `m18_jokes_dataset.csv` и достигают до **0.85 F1 (macro)** на валидации.
	•	Transformer_mix — собственная реализация на PyTorch.
	•	DeepPavlov/rubert-base-cased — адаптированная версия RuBERT.
	•	final_robert_full_training — Roberta, обученная на полном датасете

Структура проекта
Anekdots/
│
├── main.py                   # Точка входа
├── Model.py                  # Архитектура моделей
├── Аnekdoty.ipynb            # Эксперименты в Jupyter
├── anek_vocab.pkl            # Словарь токенизатора
│
├── DeepPavlov/               # Модель DeepPavlov/rubert-base-cased
│   ├── model.safetensors
│   ├── vocab.txt
│   ├── config.json
│   └── tokenizer.json
│
├── final_robert_full_training/  # Финальная Roberta модель
│   ├── model.safetensors
│   ├── vocab.json
│   ├── merges.txt
│   └── config.json
│
├── weights_*.pt              # Промежуточные веса
├── requirements.txt
└── Dockerfile

передать нужно запросом post predict (к примеру localhost:8000/predict), body - словарь 'anekdot': [анекдот или несколько]
К примеру, так:
```python
{"anekdot": [
    "Шутка про программиста, который не любит баги",
    "Черчилль спрашивает Сталина что бы вы и ваши солдаты сделали с Гитлером, будь он у вас в руках? Сталин отвечает раскалил бы кочергу докрасна и засунул бы холодным концом ему в задницу. -А почему холодным, товарищ Сталин? -Чтобы вы, господин Черчилль, не помогли ему ее вытащить",
    "Приходит пациент к доктору и говорит: доктор, хуй чешется. Доктор отвечает: мой чаще. -Нет, мой",
    "— Вовочка, ты почему не сделал домашнее задание? — А я в шахматы играл с папой."
]}
```

Ответ будет в виде предсказаний от каждой модели, к примеру
```python
{
    "Sber": {
        "Шутка про программиста, который не любит баги": "raznie",
        "Черчилль спрашивает Сталина что бы вы и ваши солдаты сделали с Гитлером, будь он у вас в руках? Сталин отвечает раскалил бы кочергу докрасна и засунул бы холодным концом ему в задницу. -А почему холодным, товарищ Сталин? -Чтобы вы, господин Черчилль, не помогли ему ее вытащить": "pro-armiu",
        "Приходит пациент к доктору и говорит: доктор, хуй чешется. Доктор отвечает: мой чаще. -Нет, мой": "meditsinskie",
        "— Вовочка, ты почему не сделал домашнее задание? — А я в шахматы играл с папой.": "pro-vovochku"
    },
    "Pavlov": {
        "Шутка про программиста, который не любит баги": "aforizmi",
        "Черчилль спрашивает Сталина что бы вы и ваши солдаты сделали с Гитлером, будь он у вас в руках? Сталин отвечает раскалил бы кочергу докрасна и засунул бы холодным концом ему в задницу. -А почему холодным, товарищ Сталин? -Чтобы вы, господин Черчилль, не помогли ему ее вытащить": "pro-armiu",
        "Приходит пациент к доктору и говорит: доктор, хуй чешется. Доктор отвечает: мой чаще. -Нет, мой": "meditsinskie",
        "— Вовочка, ты почему не сделал домашнее задание? — А я в шахматы играл с папой.": "pro-vovochku"
    },
    "Mix": {
        "Шутка про программиста, который не любит баги": "aforizmi",
        "Черчилль спрашивает Сталина что бы вы и ваши солдаты сделали с Гитлером, будь он у вас в руках? Сталин отвечает раскалил бы кочергу докрасна и засунул бы холодным концом ему в задницу. -А почему холодным, товарищ Сталин? -Чтобы вы, господин Черчилль, не помогли ему ее вытащить": "tsitati",
        "Приходит пациент к доктору и говорит: доктор, хуй чешется. Доктор отвечает: мой чаще. -Нет, мой": "tsitati",
        "— Вовочка, ты почему не сделал домашнее задание? — А я в шахматы играл с папой.": "aforizmi"
    }
}
```

Параметры моделей
Base model: Transformer_mix
Fine-tuning dataset: m18_jokes_dataset.csv
Classes: 7 thematic categories
Validation (10% split):
 • Accuracy: 0.56
 • F1 (macro): 0.56
Training: 25 epochs, batch 16, lr=1e-5, weight decay=1e-5

Base model: DeepPavlov/rubert-base-cased
Fine-tuning dataset: m18_jokes_dataset.csv
Classes: 7 thematic categories
Validation (10% split):
 • Accuracy: 0.78
 • F1 (macro): 0.77
Training: 5 epochs, batch 16, lr=1e-5, weight decay=1e-5

Base model: ./final_robert_model
Fine-tuning dataset: m18_jokes_dataset.csv
Classes: 7 thematic categories
Validation (10% split):
 • Accuracy: 0.85
 • F1 (macro): 0.85
Training: 5 epochs, batch 16, lr=1e-5, weight decay=1e-5
Final model: retrained on full dataset with same hyperparameters.

Инструкция для запуска нужна будет с учетом сборки docker контейнера

## 2. Установка и запуск сервиса

Для запуска нужно клонировать
```bash
git clone https://github.com/asphalter95/Anekdots.git
cd /Anekdots
docker build -t anekdots .
docker run -p 8000:8000 anekdots
```

Лицензия

MIT License © 2025 Petros Arakelyan