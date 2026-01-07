from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
# Enable CORS for all routes so your Vercel frontend can access this
CORS(app) 

# ---------------------------------------------------------
# DATA LOADING (Load once when server starts to save time)
# ---------------------------------------------------------
# Replace 'your_data.csv' with your actual file path
# df = pd.read_csv('your_data.csv') 

@app.route('/')
def home():
    return "API is running!"

@app.route('/api/insights', methods=['GET'])
def get_insights():
    # -----------------------------------------------------
    # YOUR PYTHON LOGIC GOES HERE
    # -----------------------------------------------------
    
    # Example: Calculate some stats (replace with real logic)
    # total_enrolment = df['enrolment_count'].sum()
    # top_state = df['state'].mode()[0]
    
    # Mock data for demonstration
    insights = {
        "total_enrolments": 1450000,
        "top_state": "Maharashtra",
        "trend": "Increasing",
        "prediction_2027": 1600000
    }
    
    return jsonify(insights)

if __name__ == '__main__':
    app.run(debug=True, port=5000)