from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load the pretrained model
model_bundle = joblib.load("decision_tree_tfidf_model.pkl")
model = model_bundle['model']
vectorizer = model_bundle['vectorizer']
 # Make sure your model is in backend folder

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
    app.run(debug=True)
