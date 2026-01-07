from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This allows your frontend to talk to this API

@app.route('/')
def hello():
    return "API is running!"

# New route to handle requests from your frontend
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json  # Get data sent from frontend
    user_message = data.get('message')

    # --- Your Logic Here ---
    # (e.g., send to Gemini/OpenAI, or just echo it back)
    response_text = f"Server received: {user_message}" 
    # -----------------------

    return jsonify({"reply": response_text})

if __name__ == '__main__':
    app.run(debug=True)