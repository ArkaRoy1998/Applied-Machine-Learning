from flask import Flask, request, render_template, jsonify
import joblib
import score

app = Flask(__name__)

# Load the trained SVM model
svm_model = joblib.load("Support_Vector_Machine_prob_final.joblib")

# Threshold for classification
threshold = 0.5
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/classify', methods=['POST'])
def classify_sentence():
    """Classify the input sentence as spam or not spam."""
    try:
        data = request.get_json()
        sentence = data.get('sentence', '')

        if not sentence:
            raise ValueError("Please provide a sentence for classification.")

        label, propensity = score.score(sentence, svm_model, threshold)
        result_label = "Spam" if label == 1 else "Not Spam"
        result_message = f'The sentence "{sentence}" is {result_label} with propensity {propensity:.2f}'

        return jsonify({'result': result_message}), 200
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return jsonify({'error': error_message}), 500

@app.route('/score', methods=['POST'])
def score_endpoint():
    return classify_sentence()
if __name__ == '__main__': # pragma no cover
    app.run(debug=True, host="0.0.0.0",use_reloader=False)
