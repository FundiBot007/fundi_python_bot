import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------- LOAD DATA ----------------
def load_data(file_path):
    questions = []
    answers = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                q, a = line.strip().split("|", 1)
                questions.append(q)
                answers.append(a)
    return questions, answers

questions, answers = load_data("data.txt")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)


# -------------- CHATBOT LOGIC ------------
def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X).flatten()
    best_match = similarities.argmax()

    if similarities[best_match] < 0.2:
        return "Sorry, I don't understand. Please rephrase."

    return answers[best_match]


# -------------- GUI SETUP ----------------
def send_message():
    user_msg = entry.get()
    chatbox.insert(tk.END, "You: " + user_msg + "\n")

    bot_msg = chatbot_response(user_msg)
    chatbox.insert(tk.END, "Bot: " + bot_msg + "\n\n")

    entry.delete(0, tk.END)


# Main window
window = tk.Tk()
window.title("My AI Chatbot")
window.geometry("400x500")

chatbox = scrolledtext.ScrolledText(window, wrap=tk.WORD)
chatbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry = tk.Entry(window, width=60)
entry.pack(padx=10, pady=5)

send_btn = tk.Button(window, text="Send", command=send_message)
send_btn.pack(pady=5)

window.mainloop()
