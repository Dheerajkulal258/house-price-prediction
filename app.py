from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    sqft = float(request.form["squarefeet"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])

    features = np.array([[sqft, bedrooms, bathrooms]])

    prediction = model.predict(features)

    output = max(0, round(prediction[0], 2))

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price: ₹ {output:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)