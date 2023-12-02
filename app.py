from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

model = joblib.load("crime.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.method == "POST":
            
            state = float(request.form["state"])
            district = float(request.form["district"])
            year = int(request.form["year"])
            
        

            inputs = pd.DataFrame([[state, district, year]], columns=[['STATE/UT', 'DISTRICT', 'YEAR']])
            ans = model.predict(inputs)[0] # Access the first element as the result is a single prediction

        return render_template('result.html', ans=ans)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == "__main__":
    app.run()