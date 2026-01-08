import pickle

model = pickle.load(open("final_model.sav", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

text = input("Enter the news statement: ")

vec = vectorizer.transform([text])
prediction = model.predict(vec)
prob = model.predict_proba(vec)

print("\nPrediction:", "REAL" if prediction[0] == 1 else "FAKE")
print("Truth Probability:", round(prob[0][1], 2))
