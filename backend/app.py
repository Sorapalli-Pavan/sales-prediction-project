from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model
try:
    model = pickle.load(open('backend/model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        TV = data.get('TV', 0)
        Radio = data.get('Radio', 0)
        Newspaper = data.get('Newspaper', 0)

        features = np.array([[TV, Radio, Newspaper]])
        prediction = model.predict(features)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-data', methods=['GET'])
def get_data():
    try:
        data = pd.read_csv('backend/data/advertising.csv')
        return data.to_dict(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
