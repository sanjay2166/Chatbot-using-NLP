import random
import nltk
import numpy as np
import string
import datetime
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

API_KEY = "your_openweather_api_key"

intents = [
    {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
     "responses": ["Hi there!", "Hello!", "Hey! How can I help?", "I'm doing well, thanks for asking."]},
    {"tag": "goodbye", "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
     "responses": ["Goodbye!", "See you soon!", "Take care!"]},
    {"tag": "thanks", "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
     "responses": ["You're welcome!", "No problem!", "Glad I could help!"]},
    {"tag": "about", "patterns": ["What can you do", "Who are you", "What is your purpose"],
     "responses": ["I'm a chatbot designed to assist you with queries and have friendly conversations!"]},
    {"tag": "time",
     "patterns": ["What time is it?", "Tell me the current time", "What's the time now?", "Current time?"],
     "responses": []},
    {"tag": "weather", "patterns": ["What's the weather like?", "Tell me the weather", "Current weather?"],
     "responses": []},
    {"tag": "calculate", "patterns": ["Calculate", "Solve", "What is"],
     "responses": []}
]


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens])


training_sentences, training_labels = [], []
tag_to_response = {}

for intent in intents:
    tag_to_response[intent['tag']] = intent['responses']
    for pattern in intent['patterns']:
        training_sentences.append(preprocess_text(pattern))
        training_labels.append(intent['tag'])

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_sentences)
label_encoder = {tag: idx for idx, tag in enumerate(tag_to_response.keys())}
y_train = np.array([label_encoder[label] for label in training_labels])

model = LogisticRegression()
model.fit(X_train, y_train)


def get_weather():
    city = "Your_City"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()
    if response.get("main"):
        temperature = response["main"]["temp"]
        weather_desc = response["weather"][0]["description"]
        return f"The current temperature in {city} is {temperature}°C with {weather_desc}."
    return "Sorry, I couldn't fetch the weather right now."


def calculate_expression(expression):
    try:
        result = eval(expression)
        return f"The result is {result}."
    except:
        return "Sorry, I couldn't calculate that."


def chatbot_response(user_input):
    user_input_processed = preprocess_text(user_input)
    input_vector = vectorizer.transform([user_input_processed])
    prediction = model.predict(input_vector)[0]
    predicted_tag = list(label_encoder.keys())[list(label_encoder.values()).index(prediction)]

    if predicted_tag == "time":
        return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."
    elif predicted_tag == "weather":
        return get_weather()
    elif predicted_tag == "calculate":
        return calculate_expression(user_input.replace('calculate', '').strip())

    response = random.choice(tag_to_response.get(predicted_tag, ["I'm not sure how to respond to that."]))
    return response


print("Chatbot is running! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye!")
        break
    print(f"Chatbot: {chatbot_response(user_input)}")
