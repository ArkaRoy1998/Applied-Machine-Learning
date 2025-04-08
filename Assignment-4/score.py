import joblib
from sentence_transformers import SentenceTransformer


class SentimentClassifier:
    """
    A sentiment classifier based on a trained model and sentence embeddings.

    Attributes:
        model_path (str): Path to the trained model file.
        model: The loaded trained model.
        sentence_encoder: Sentence transformer model for encoding text.
    """

    def __init__(self, model_path: str):
        """
        Initialize the SentimentClassifier.

        Args:
            model_path (str): Path to the trained model file.
        """
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def score_sentiment(self, text: str, threshold: float = 0.5) -> (bool, float):
        try:
            # Transform the input text using the sentence encoder.
            embedding = self.sentence_encoder.encode([text])

            # Compute the probability (propensity) for the positive class.
            propensity = self.model.predict_proba(embedding)[:, 1][0]

            # Apply the threshold: if propensity is greater than or equal to the threshold, assign label 1; else 0.
            prediction = 1 if propensity >= threshold else 0

            return prediction, propensity
        except Exception as e:
            raise ValueError(f"Error scoring sentiment: {str(e)}")


classifier = SentimentClassifier("Support_Vector_Machine_prob_final.joblib")

def score(sentence, model=None, threshold=0.5):
    return classifier.score_sentiment(sentence, threshold)


if __name__ == "__main__": # pragma no cover
    try:
        classifier = SentimentClassifier("Support_Vector_Machine_prob_final.joblib")
        text = "This is a great product!"
        label, propensity = classifier.score_sentiment(text)
        sentiment = "Positive" if label else "Negative"
        print(f"The sentiment of '{text}' is {sentiment} with a propensity score of {propensity:.2f}")
    except ValueError as ve:
        print(ve)
