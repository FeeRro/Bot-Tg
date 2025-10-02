# Создадим простой пример нейросети для чат-бота обучения английскому языку
import numpy as np
import pandas as pd
import json

# Создадим структуру данных для обучающих материалов по английскому языку
training_data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["hello", "hi", "hey", "good morning", "good afternoon"],
            "responses": ["Hello! Let's practice English together!", "Hi there! Ready for an English lesson?", "Hey! What would you like to learn today?"],
            "context": "greeting"
        },
        {
            "tag": "grammar_question",
            "patterns": ["what is present tense", "explain past tense", "help with grammar", "grammar rules"],
            "responses": ["Present tense describes actions happening now. For example: 'I eat breakfast every morning.'", "Let me help you with grammar! What specific topic interests you?"],
            "context": "grammar"
        },
        {
            "tag": "vocabulary",
            "patterns": ["teach me words", "new words", "vocabulary lesson", "word meaning"],
            "responses": ["Let's learn new words! Here's a word: 'Beautiful' means attractive or pleasing to look at.", "Vocabulary time! Try using this word in a sentence: 'Adventure'"],
            "context": "vocabulary"
        },
        {
            "tag": "practice",
            "patterns": ["I want to practice", "let's practice", "practice speaking", "conversation practice"],
            "responses": ["Great! Let's have a conversation. Tell me about your day.", "Perfect! Practice makes perfect. What topic would you like to talk about?"],
            "context": "practice"
        }
    ]
}

# Сохраним данные в JSON файл
with open('english_learning_intents.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

print("Создан файл с обучающими данными для английского языка")
print(f"Количество интентов: {len(training_data['intents'])}")

# Создадим простую нейронную сеть с использованием только numpy
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Инициализация весов
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
    
    def sigmoid(self, x):
        """Сигмоидная функция активации"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Производная сигмоидной функции"""
        return x * (1 - x)
    
    def forward(self, X):
        """Прямое распространение"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        """Обратное распространение"""
        m = X.shape[0]
        
        # Вычисление ошибки на выходном слое
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Вычисление ошибки на скрытом слое
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Обновление весов
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs):
        """Обучение нейронной сети"""
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """Предсказание"""
        return self.forward(X)

# Создадим простые данные для демонстрации
def create_sample_data():
    """Создание примерных данных для обучения"""
    # Простые паттерны для классификации типов вопросов
    patterns = [
        [1, 0, 0, 1, 0],  # greeting
        [0, 1, 1, 0, 0],  # grammar
        [0, 0, 1, 1, 1],  # vocabulary  
        [1, 1, 0, 0, 1],  # practice
    ]
    
    labels = [
        [1, 0, 0, 0],  # greeting
        [0, 1, 0, 0],  # grammar
        [0, 0, 1, 0],  # vocabulary
        [0, 0, 0, 1],  # practice
    ]
    
    return np.array(patterns), np.array(labels)

# Создание и обучение нейронной сети
X, y = create_sample_data()
print(f"\nДанные для обучения:")
print(f"Входные данные: {X.shape}")
print(f"Выходные данные: {y.shape}")

# Инициализация нейронной сети
nn = SimpleNeuralNetwork(input_size=5, hidden_size=8, output_size=4, learning_rate=0.5)

print("\nОбучение нейронной сети...")
losses = nn.train(X, y, epochs=1000)

# Тестирование нейронной сети
print("\nТестирование модели:")
test_input = np.array([[1, 0, 0, 1, 0]])  # Тестовый вход (приветствие)
prediction = nn.predict(test_input)
predicted_class = np.argmax(prediction)

class_names = ["greeting", "grammar", "vocabulary", "practice"]
print(f"Входные данные: {test_input[0]}")
print(f"Предсказания: {prediction[0]}")
print(f"Предсказанный класс: {class_names[predicted_class]}")

print(f"\nФинальная ошибка: {losses[-1]:.4f}")