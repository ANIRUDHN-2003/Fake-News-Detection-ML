from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("final_model.sav", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    probability = ""

    if request.method == "POST":
        text = request.form["news"]
        vec = vectorizer.transform([text])
        result = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]

        prediction = "REAL" if result == 1 else "FAKE"
        probability = round(prob, 2)

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
