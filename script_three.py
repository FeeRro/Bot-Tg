# Создадим расширенный набор данных для обучения
extended_training_data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "howdy", "greetings", "what's up", "sup", "yo", "nice to meet you"
            ],
            "responses": [
                "Hello! Let's practice English together! 🇬🇧",
                "Hi there! Ready for an English lesson? 📚", 
                "Hey! What would you like to learn today? 🎯",
                "Greetings! I'm here to help you improve your English! 🌟",
                "Hello! Welcome to your English learning journey! 🚀"
            ],
            "context": "greeting"
        },
        {
            "tag": "grammar_question",
            "patterns": [
                "what is present tense", "explain past tense", "help with grammar", 
                "grammar rules", "teach me tenses", "how to use articles",
                "when to use 'a' or 'an'", "explain future tense", "passive voice",
                "present perfect", "past perfect", "conditional sentences"
            ],
            "responses": [
                "Present tense describes actions happening now. For example: 'I eat breakfast every morning.' 🍳",
                "Past tense is for completed actions. Example: 'I visited London last year.' 🏰",
                "Let me help you with grammar! What specific topic interests you? 📖",
                "Grammar is the foundation of language! Ask me about any tense or rule. 🏗️",
                "Great question! Grammar makes your English clear and correct. What would you like to know? ✨"
            ],
            "context": "grammar"
        },
        {
            "tag": "vocabulary",
            "patterns": [
                "teach me words", "new words", "vocabulary lesson", "word meaning",
                "synonyms", "antonyms", "difficult words", "expand my vocabulary",
                "word of the day", "learn new vocabulary", "improve my vocabulary"
            ],
            "responses": [
                "Let's learn new words! Here's a word: 'Magnificent' means extremely beautiful or impressive. 🌟",
                "Vocabulary time! Try using this word in a sentence: 'Serendipity' - a pleasant surprise. 🎁",
                "New word alert! 'Perseverance' means continuing despite difficulties. Keep going! 💪",
                "Building vocabulary is key to fluency! Here's a useful word: 'Eloquent' - speaking fluently. 🗣️"
            ],
            "context": "vocabulary"
        },
        {
            "tag": "practice",
            "patterns": [
                "I want to practice", "let's practice", "practice speaking", 
                "conversation practice", "help me speak", "practice English",
                "let's talk", "conversation", "speaking practice", "practice dialogue"
            ],
            "responses": [
                "Great! Let's have a conversation. Tell me about your day. 🌅",
                "Perfect! Practice makes perfect. What topic would you like to talk about? 💬",
                "Excellent! Let's practice. Describe your favorite place to visit. 🏖️",
                "Wonderful! Speaking practice is so important. Tell me about your hobbies. 🎨"
            ],
            "context": "practice"
        },
        {
            "tag": "writing_help",
            "patterns": [
                "help with writing", "writing practice", "essay help", "improve my writing",
                "how to write", "writing tips", "composition help", "paragraph writing"
            ],
            "responses": [
                "Writing is a great skill to develop! Start with simple sentences and build up. ✍️",
                "For good writing, remember: clear structure, good vocabulary, and correct grammar! 📝",
                "Practice writing daily! Try describing what you see around you. 👀",
                "Writing tip: Read your work aloud to check if it sounds natural! 🔊"
            ],
            "context": "writing"
        },
        {
            "tag": "pronunciation",
            "patterns": [
                "pronunciation help", "how to pronounce", "speaking correctly",
                "accent help", "pronunciation practice", "speak clearly"
            ],
            "responses": [
                "Pronunciation improves with practice! Try repeating words slowly first. 🗣️",
                "Listen to native speakers and imitate their pronunciation. Practice daily! 🎧",
                "Focus on difficult sounds first. Break words into syllables: beau-ti-ful. 📢",
                "Record yourself speaking and compare with native speakers! 🎤"
            ],
            "context": "pronunciation"
        },
        {
            "tag": "listening",
            "patterns": [
                "listening practice", "improve listening", "understand English better",
                "listening skills", "comprehension help", "hearing English"
            ],
            "responses": [
                "For better listening, start with slower content and gradually increase speed! 👂",
                "Watch English movies with subtitles, then without! 🎬",
                "Listen to English podcasts about topics you enjoy! 🎙️",
                "Practice active listening - try to summarize what you heard! 📻"
            ],
            "context": "listening"
        },
        {
            "tag": "encouragement",
            "patterns": [
                "I'm struggling", "this is difficult", "I can't learn", "too hard",
                "want to give up", "not improving", "frustrated", "discouraged"
            ],
            "responses": [
                "Don't give up! Every expert was once a beginner. You're doing great! 💪",
                "Learning a language takes time. Celebrate small victories! 🎉",
                "Mistakes are part of learning. Keep practicing - you'll improve! 🌱",
                "Remember why you started. Your persistence will pay off! ⭐",
                "It's normal to feel challenged. That means you're growing! 🌿"
            ],
            "context": "motivation"
        },
        {
            "tag": "goodbye",
            "patterns": [
                "bye", "goodbye", "see you later", "talk to you later", "farewell",
                "catch you later", "until next time", "see ya", "take care"
            ],
            "responses": [
                "Goodbye! Keep practicing your English! See you soon! 👋",
                "Take care! Remember to practice a little every day! 🌟",
                "See you later! Keep up the great work with your English! 📚",
                "Farewell! Your English journey continues - stay motivated! 🚀"
            ],
            "context": "farewell"
        }
    ]
}

# Сохраняем расширенные данные
with open('extended_english_intents.json', 'w', encoding='utf-8') as f:
    json.dump(extended_training_data, f, indent=2, ensure_ascii=False)

print("Создан расширенный набор данных для обучения")
print(f"Количество интентов: {len(extended_training_data['intents'])}")

# Подсчитаем статистику
total_patterns = sum(len(intent['patterns']) for intent in extended_training_data['intents'])
total_responses = sum(len(intent['responses']) for intent in extended_training_data['intents'])

print(f"Общее количество паттернов: {total_patterns}")
print(f"Общее количество ответов: {total_responses}")

# Создадим таблицу статистики по интентам
stats_data = []
for intent in extended_training_data['intents']:
    stats_data.append({
        'Intent': intent['tag'],
        'Patterns': len(intent['patterns']),
        'Responses': len(intent['responses']),
        'Context': intent['context']
    })

stats_df = pd.DataFrame(stats_data)
print("\nСтатистика по интентам:")
print(stats_df.to_string(index=False))

# Сохраним статистику в CSV
stats_df.to_csv('intents_statistics.csv', index=False, encoding='utf-8')
print("\nСохранена статистика в файл intents_statistics.csv")

# Создадим пример использования нейросети с расширенными данными
demo_code = '''
# Демонстрация работы нейросети с расширенными данными

import numpy as np
import json

# Загружаем расширенные данные
with open('extended_english_intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Создаем словарь
vocab = set()
for intent in data['intents']:
    for pattern in intent['patterns']:
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
    "I want to practice speaking"
]

print("\\nПримеры векторизации:")
for phrase in test_phrases:
    vector = text_to_vector(phrase, vocab)
    active_words = [vocab[i] for i, val in enumerate(vector) if val == 1]
    print(f"Фраза: '{phrase}'")
    print(f"Активные слова: {active_words}")
    print(f"Размер вектора: {len(vector)}, Активных элементов: {int(sum(vector))}\\n")
'''

with open('demo_neural_network.py', 'w', encoding='utf-8') as f:
    f.write(demo_code)

print("Создан файл demo_neural_network.py с демонстрацией")