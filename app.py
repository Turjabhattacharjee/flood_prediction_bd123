from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and dataset
model = joblib.load("model.pkl")
df = pd.read_csv("dataset.csv")

station_index_map = dict(zip(df['Station'], df['StationIndex']))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    city = data.get("city").capitalize()
    date_str = data.get("date")
    
    if city not in station_index_map:
        return jsonify({"error": "City not found."}), 400
    
    date = datetime.strptime(date_str, "%Y-%m-%d")
    avg_rainfall = df[df['Station'] == city]['Rainfall'].mean()
    
    input_data = pd.DataFrame([[date.year, date.month, date.day, avg_rainfall, station_index_map[city]]],
                              columns=["Year", "Month", "Day", "Rainfall", "StationIndex"])
    
    flood_probability = model.predict_proba(input_data)[:, 1][0]
    return jsonify({"city": city, "date": date_str, "flood_probability": flood_probability})

if __name__ == "__main__":
    app.run(debug=True)
