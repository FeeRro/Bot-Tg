# Создадим полный пример Telegram-бота для обучения английскому языку
telegram_bot_code = '''
import telebot
import numpy as np
import json
import random
import pandas as pd
from datetime import datetime

# Класс простой нейронной сети (тот же, что мы создали выше)
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def predict(self, X):
        return self.forward(X)

# Класс для обработки английского языка
class EnglishLearningBot:
    def __init__(self):
        # Загружаем интенты
        with open('english_learning_intents.json', 'r', encoding='utf-8') as f:
            self.intents = json.load(f)
        
        # Создаем словарь для векторизации
        self.vocab = self.create_vocabulary()
        
        # Инициализируем нейронную сеть
        self.model = SimpleNeuralNetwork(
            input_size=len(self.vocab),
            hidden_size=16,
            output_size=len(self.intents['intents'])
        )
        
        # Обучаем модель
        self.train_model()
    
    def create_vocabulary(self):
        """Создание словаря из всех паттернов"""
        vocab = set()
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                words = pattern.lower().split()
                vocab.update(words)
        return sorted(list(vocab))
    
    def text_to_vector(self, text):
        """Преобразование текста в вектор (bag of words)"""
        words = text.lower().split()
        vector = np.zeros(len(self.vocab))
        for word in words:
            if word in self.vocab:
                vector[self.vocab.index(word)] = 1
        return vector
    
    def train_model(self):
        """Обучение модели на данных интентов"""
        X_train = []
        y_train = []
        
        for i, intent in enumerate(self.intents['intents']):
            for pattern in intent['patterns']:
                vector = self.text_to_vector(pattern)
                X_train.append(vector)
                
                # One-hot encoding для класса
                label = np.zeros(len(self.intents['intents']))
                label[i] = 1
                y_train.append(label)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Обучение модели
        for epoch in range(500):
            output = self.model.forward(X_train)
            loss = np.mean((output - y_train) ** 2)
            
            # Простое обновление весов
            m = X_train.shape[0]
            dZ2 = output - y_train
            dW2 = (1/m) * np.dot(self.model.a1.T, dZ2)
            db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
            
            dZ1 = np.dot(dZ2, self.model.W2.T) * (self.model.a1 * (1 - self.model.a1))
            dW1 = (1/m) * np.dot(X_train.T, dZ1)
            db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
            
            self.model.W2 -= 0.5 * dW2
            self.model.b2 -= 0.5 * db2
            self.model.W1 -= 0.5 * dW1
            self.model.b1 -= 0.5 * db1
    
    def get_response(self, message):
        """Получение ответа на сообщение пользователя"""
        vector = self.text_to_vector(message)
        prediction = self.model.predict(np.array([vector]))
        intent_index = np.argmax(prediction)
        
        # Проверяем уверенность модели
        confidence = np.max(prediction)
        if confidence < 0.7:
            return "I'm not sure I understand. Could you please rephrase your question?"
        
        intent = self.intents['intents'][intent_index]
        return random.choice(intent['responses'])

# Словари с учебными материалами
vocabulary_lessons = {
    'beginner': [
        {'word': 'hello', 'meaning': 'a greeting used when meeting someone', 'example': 'Hello, how are you?'},
        {'word': 'book', 'meaning': 'a set of printed sheets bound together', 'example': 'I am reading a book.'},
        {'word': 'water', 'meaning': 'a clear liquid that forms seas, lakes, rivers', 'example': 'I drink water every day.'},
    ],
    'intermediate': [
        {'word': 'adventure', 'meaning': 'an exciting or unusual experience', 'example': 'Our trip was a great adventure.'},
        {'word': 'knowledge', 'meaning': 'information and skills acquired through experience', 'example': 'Knowledge is power.'},
    ]
}

grammar_lessons = {
    'present_simple': {
        'rule': 'Used for habits, facts, and general truths',
        'structure': 'Subject + Verb (base form) + Object',
        'examples': ['I eat breakfast every morning.', 'She works in an office.', 'The sun rises in the east.']
    },
    'past_simple': {
        'rule': 'Used for completed actions in the past',
        'structure': 'Subject + Verb (past form) + Object',
        'examples': ['I visited Paris last year.', 'She finished her homework.', 'They played football yesterday.']
    }
}

# Основной код бота
class TelegramEnglishBot:
    def __init__(self, token):
        self.bot = telebot.TeleBot(token)
        self.english_bot = EnglishLearningBot()
        self.user_progress = {}  # Хранение прогресса пользователей
        
        # Регистрируем обработчики
        self.register_handlers()
    
    def register_handlers(self):
        @self.bot.message_handler(commands=['start'])
        def start_message(message):
            welcome_text = """
🇬🇧 Welcome to English Learning Bot! 🇬🇧

I'm here to help you learn English through:
📚 Vocabulary lessons
📝 Grammar explanations  
💬 Conversation practice
🎯 Interactive exercises

Use these commands:
/vocabulary - Learn new words
/grammar - Grammar lessons
/practice - Conversation practice
/progress - Check your progress

Just type anything to start chatting!
            """
            self.bot.send_message(message.chat.id, welcome_text)
            
            # Инициализация прогресса пользователя
            if message.from_user.id not in self.user_progress:
                self.user_progress[message.from_user.id] = {
                    'level': 'beginner',
                    'lessons_completed': 0,
                    'words_learned': [],
                    'last_activity': datetime.now()
                }
        
        @self.bot.message_handler(commands=['vocabulary'])
        def vocabulary_lesson(message):
            user_id = message.from_user.id
            level = self.user_progress.get(user_id, {}).get('level', 'beginner')
            
            if level in vocabulary_lessons:
                lesson = random.choice(vocabulary_lessons[level])
                response = f"""
📚 Vocabulary Lesson:

Word: *{lesson['word']}*
Meaning: {lesson['meaning']}
Example: _{lesson['example']}_

Try using this word in your own sentence!
                """
                self.bot.send_message(message.chat.id, response, parse_mode='Markdown')
                
                # Обновляем прогресс
                if user_id in self.user_progress:
                    self.user_progress[user_id]['words_learned'].append(lesson['word'])
        
        @self.bot.message_handler(commands=['grammar'])
        def grammar_lesson(message):
            lesson_type = random.choice(list(grammar_lessons.keys()))
            lesson = grammar_lessons[lesson_type]
            
            response = f"""
📝 Grammar Lesson: {lesson_type.replace('_', ' ').title()}

Rule: {lesson['rule']}
Structure: {lesson['structure']}

Examples:
"""
            for example in lesson['examples']:
                response += f"• {example}\\n"
            
            self.bot.send_message(message.chat.id, response)
        
        @self.bot.message_handler(commands=['practice'])
        def practice_session(message):
            practice_questions = [
                "Tell me about your favorite hobby.",
                "What did you do yesterday?",
                "Describe your ideal weekend.",
                "What are your plans for tomorrow?",
                "Tell me about your family."
            ]
            
            question = random.choice(practice_questions)
            response = f"🗣️ Practice Time!\\n\\n{question}\\n\\nTake your time and answer in English!"
            self.bot.send_message(message.chat.id, response)
        
        @self.bot.message_handler(commands=['progress'])
        def show_progress(message):
            user_id = message.from_user.id
            progress = self.user_progress.get(user_id, {})
            
            if not progress:
                self.bot.send_message(message.chat.id, "Start learning first! Use /start to begin.")
                return
            
            response = f"""
🎯 Your Learning Progress:

Level: {progress.get('level', 'beginner').title()}
Lessons completed: {progress.get('lessons_completed', 0)}
Words learned: {len(progress.get('words_learned', []))}
Last activity: {progress.get('last_activity', 'Never').strftime('%Y-%m-%d %H:%M') if isinstance(progress.get('last_activity'), datetime) else 'Never'}

Keep up the great work! 🌟
            """
            self.bot.send_message(message.chat.id, response)
        
        @self.bot.message_handler(func=lambda message: True)
        def handle_message(message):
            # Используем нейронную сеть для обработки сообщения
            response = self.english_bot.get_response(message.text)
            
            # Обновляем активность пользователя
            user_id = message.from_user.id
            if user_id in self.user_progress:
                self.user_progress[user_id]['last_activity'] = datetime.now()
            
            self.bot.send_message(message.chat.id, response)
    
    def run(self):
        print("English Learning Bot запущен!")
        self.bot.polling(none_stop=True)

# Пример использования (нужно будет заменить YOUR_BOT_TOKEN на реальный токен)
if __name__ == "__main__":
    # TOKEN = "YOUR_BOT_TOKEN_HERE"  # Получить у @BotFather
    # bot = TelegramEnglishBot(TOKEN)
    # bot.run()
    print("Код бота готов! Для запуска нужно получить токен у @BotFather")
'''

# Сохраняем код бота в файл
with open('english_learning_telegram_bot.py', 'w', encoding='utf-8') as f:
    f.write(telegram_bot_code)

print("Создан полный код Telegram-бота для обучения английскому языку!")
print("Файл: english_learning_telegram_bot.py")

# Создадим дополнительный файл с requirements
requirements = '''numpy==1.21.0
pandas==1.3.0
pyTelegramBotAPI==4.0.0
'''

with open('requirements.txt', 'w') as f:
    f.write(requirements)

print("Создан файл requirements.txt с зависимостями")

# Создадим README с инструкциями
readme_content = '''# English Learning Telegram Bot with Neural Network

Это Telegram-бот для изучения английского языка, использующий простую нейронную сеть на NumPy и pandas.

## Возможности бота:
- 🧠 Использует собственную нейронную сеть для понимания запросов
- 📚 Уроки словарного запаса
- 📝 Объяснения грамматики  
- 💬 Практика разговора
- 🎯 Отслеживание прогресса

## Установка:

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Получите токен бота у @BotFather в Telegram

3. Замените YOUR_BOT_TOKEN_HERE в коде на ваш токен

4. Запустите бота:
```bash
python english_learning_telegram_bot.py
```

## Команды бота:
- `/start` - Начать работу с ботом
- `/vocabulary` - Урок словарного запаса
- `/grammar` - Урок грамматики
- `/practice` - Практика разговора
- `/progress` - Показать прогресс обучения

## Архитектура:
- **SimpleNeuralNetwork**: Класс нейронной сети с двумя слоями
- **EnglishLearningBot**: Обработка NLP и интентов
- **TelegramEnglishBot**: Основной класс Telegram-бота

## Файлы:
- `english_learning_telegram_bot.py` - Основной код бота
- `english_learning_intents.json` - Данные для обучения нейросети
- `requirements.txt` - Зависимости проекта
'''

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("Создан файл README.md с инструкциями")
print("\nВсе файлы проекта готовы!")