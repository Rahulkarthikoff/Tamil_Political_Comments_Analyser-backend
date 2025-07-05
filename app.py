from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load the pretrained model
model_bundle = joblib.load("decision_tree_tfidf_model.pkl")
model = model_bundle['model']
vectorizer = model_bundle['vectorizer']
 # Make sure your model is in backend folder

@app.route('/')
def home():
    return 'Tamil Sentiment Analysis API is running.'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No input text provided'}), 400
    
    X = vectorizer.transform([text])

    prediction = model.predict(X)[0]
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
