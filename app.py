import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Specify a custom path for NLTK data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Add the custom directory to NLTK's data path
nltk.data.path.append(nltk_data_path)

# Download the 'punkt' tokenizer if not already present
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
    nltk.download('punkt', download_dir=nltk_data_path)

# This will bypass SSL verification, which is necessary for some environments
ssl._create_default_https_context = ssl._create_unverified_context

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:  
    intents = json.load(file)

vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = input_text.lower().strip()
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter
    st.title("MediBot")
    st.write("**Medicine Information Chatbot**")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("""
Welcome to **MediBot**! I'm your friendly neighborhood chatbot with a brain full of medicine knowledge. Need some quick info about your meds? Just ask away! 

I can help you with details like:

- **Usage** – What’s this medicine for?
- **Side effects** – What could go wrong? (But don’t worry, I’ll keep it light!)
- **Dosage** – How much should you take? (No need for math – I’ve got you covered!)

Just type the name of any medication, and I’ll do the rest! Let’s get your health knowledge up to speed!
# #Please make sure to provide the exact name of the medication. If you need more assistance, refer to the 'About' section in the menu.##
""")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()


    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that provides accurate and reliable information about various medications, including their uses, side effects, dosages, and compositions. The chatbot is built using Natural Language Processing (NLP) techniques and Logistic Regression to identify user intents and extract relevant entities from the user's queries. Streamlit, a Python web framework, is used to build an interactive web-based chatbot interface, making it easy for users to get medication-related information quickly and efficiently.")

        st.subheader("Project Overview:")

        st.write("""
            Key Components:

            NLP and Logistic Regression:

            1.The chatbot uses NLP techniques to analyze user input and categorize it into predefined intents, such as asking about specific medications or seeking general information.
            Logistic Regression is employed to classify and map user input to the appropriate intent and response.
            Streamlit Interface:

            2.Streamlit is used to create an interactive and user-friendly interface where users can type their questions and receive responses from the chatbot.
            The chatbot provides an interface where users can interact with it by entering text, and it responds with information based on the intent and entities extracted.
        """)

        st.subheader("Dataset:")

        st.write("""
        The dataset consists of a collection of labeled intents and patterns related to different medicines. Each intent has predefined patterns (user queries) and corresponding responses.

        Here are some key intents and their examples:

        Greeting:

        Patterns: ["Hi", "Hello", "Hey", "How are you"]
        Responses: ["Hello! How can I assist you with your medicine queries today?", "Hi there! What medicine information do you need?", "Hey! How can I help you today?"]
        Goodbye:

        Patterns: ["Bye", "See you later", "Goodbye", "Take care"]
        Responses: ["Goodbye! Take care of your health!", "See you later! Stay healthy!", "Goodbye! Feel free to reach out if you have more questions!"]
        Thanks:

        Patterns: ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"]
        Responses: ["You're welcome! Stay healthy!", "No problem! Glad I could help!", "You're welcome! Take care!"]
        About:

        Patterns: ["What can you do", "Who are you", "What are you", "What is your purpose"]
        Responses: ["I am a medicine information chatbot here to provide details about various medications.", "My purpose is to assist you with information about medicines and their uses."]
        Medicine Information:

        For each medicine, such as Paracetamol, Ibuprofen, Amoxicillin, Metformin, and Atorvastatin, the chatbot identifies specific queries and provides detailed information on:
        Usage
        Side effects
        Dosage

        """)

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")
        st.write("""
                This Medicine Bot is a simple chatbot designed to provide basic and relevant information about some of the most commonly used medicines. The following are the medicines included in the bot's dataset:

                - Paracetamol (Acetaminophen)
                - Metformin
                - Ibuprofen
                - Amlodipine
                - Atorvastatin
                - Pantoprazole
                - Amoxicillin
                - Losartan
                - Omeprazole
                - Azithromycin
                - Dolo 650
                - Colpol 500
                - Meftal-Spas
                - Alerid

                Please note that this bot provides basic information and is not a substitute for professional medical advice.
                """)


        st.write("This project aims to create a user-friendly, accessible chatbot that offers users relevant, accurate medicine-related information. The combination of NLP, Logistic Regression, and Streamlit allows for both efficient classification of intents and an interactive interface for user queries. The chatbot is a great starting point and can be extended further with additional data, more advanced NLP models, or deeper learning techniques")
if __name__ == '__main__':
    main()
