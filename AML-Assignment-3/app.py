from flask import Flask, request, render_template
import joblib
import score
import numpy as np
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# Load resources once at startup
MODEL_PATH = "/Users/arkaroy/Downloads/Support_Vector_Machine_prob_final.joblib"
SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Load classification model
try: # pragma no cover
    print("Loading classification model...")
    with open(MODEL_PATH, 'rb') as model_file:
        CLASSIFIER_MODEL = joblib.load(model_file)
    print("Model loaded successfully")
except Exception as e: # pragma no cover
    print(f"Error loading model: {str(e)}")
    raise

THRESHOLD = 0.5


@app.route('/')
def home():
    """Render the home page."""
    return render_template('spam.html')


@app.route('/spam', methods=['POST'])
def spam():
    """Handle the spam classification request."""
    try:
        sentence = request.form['sent']
        embedding = SENTENCE_MODEL.encode([sentence])[0]
        label, propensity = score.score(embedding, CLASSIFIER_MODEL, THRESHOLD)

        label_text = "Spam" if label else "Not spam"
        response_message = f'The sentence "{sentence}" is {label_text} with propensity {propensity:.4f}'

        return render_template('res.html', ans=response_message)

    except KeyError:
        return render_template('error.html', error="Missing 'sent' field in form data"), 400
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


if __name__ == '__main__': #  pragma: no cover
    print("Starting Flask server...")
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)
    finally:
        print("Server stopped")