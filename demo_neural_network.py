# Демонстрация работы нейросети с расширенными данными

import numpy as np
import json

# Загружаем расширенные данные
with open("extended_english_intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Создаем словарь
vocab = set()
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words = pattern.lower().split()
        vocab.update(words)

vocab = sorted(list(vocab))
print(f"Словарь содержит {len(vocab)} уникальных слов")


# Примеры векторизации
def text_to_vector(text, vocabulary):
    words = text.lower().split()
    vector = np.zeros(len(vocabulary))
    for word in words:
        if word in vocabulary:
            vector[vocabulary.index(word)] = 1
    return vector


# Тестовые фразы
test_phrases = [
    "hello how are you",
    "help me with grammar",
    "teach me new words",
    "I want to practice speaking",
]

print("\nПримеры векторизации:")
for phrase in test_phrases:
    vector = text_to_vector(phrase, vocab)
    active_words = [vocab[i] for i, val in enumerate(vector) if val == 1]
    print(f"Фраза: '{phrase}'")
    print(f"Активные слова: {active_words}")
    print(f"Размер вектора: {len(vector)}, Активных элементов: {int(sum(vector))}\n")
