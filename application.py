import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template

application = Flask(__name__)
app = application

# import ElasticNet Regressor and Standard Scaler pickle
ridge_model = pickle.load(open("models/elastic_net_model_algerian_forestfire.pkl", "rb"))
scaler = pickle.load(open("models/standard_scaler.pkl", "rb"))

@app.route("/predictData", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        temp = float(request.form.get("Temperature"))
        rh = float(request.form.get("RH"))
        ws = float(request.form.get("Ws"))
        rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        DC = float(request.form.get("DC"))

        scaled_data = scaler.transform([[temp, rh, ws, rain, FFMC, DMC, DC]])

        result = ridge_model.predict(scaled_data)

        return render_template("home.html", results=result[0])
                
    else:
        return render_template("home.html")


@app.route("/")
def index():
    return render_template("index.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")
