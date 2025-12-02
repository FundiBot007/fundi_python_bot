from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -------------------------
# LOAD DATA
# -------------------------
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

# -------------------------
# CHAT FUNCTION
# -------------------------
def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X).flatten()
    best_match = similarities.argmax()
    if similarities[best_match] < 0.2:
        return "Sorry, I don't understand. Please rephrase."
    return answers[best_match]

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["message"]
    response = chatbot_response(user_input)
    return response

if __name__ == "__main__":
    app.run(debug=True)