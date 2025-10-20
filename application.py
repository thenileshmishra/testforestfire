from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os

application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickle
# Build paths relative to this script so the app can be started from any CWD
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'

ridge_path = MODELS_DIR / 'ridge.pkl'
scaler_path = MODELS_DIR / 'scaler.pkl'

if not ridge_path.exists() or not scaler_path.exists():
    raise FileNotFoundError(f"Required model files not found in {MODELS_DIR}.\n"
                            f"Expected: {ridge_path} and {scaler_path}")

ridge_model = pickle.load(open(ridge_path, 'rb'))
standard_scaler = pickle.load(open(scaler_path, 'rb'))




@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html', results=result[0])
    else:
        return render_template("home.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)