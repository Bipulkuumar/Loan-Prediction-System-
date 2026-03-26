from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model_path = os.path.join("model", "loan_approval_model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        int(request.form["cibil"]),
        int(request.form["income"]),
        int(request.form["employment"]),
        int(request.form["loan_amount"]),
        int(request.form["tenure"]),
        int(request.form["avg_balance"]),
        int(request.form["emi"]),
        int(request.form["transactions"])
    ]

    final_features = np.array([features])
    prediction = model.predict(final_features)[0]

    result = "LOAN APPROVED ✅" if prediction == 1 else "LOAN REJECTED ❌"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
