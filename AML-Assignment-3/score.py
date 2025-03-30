import joblib
import sys
import numpy as np
from typing import Tuple
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator
from typing import Tuple
def score(embedding: np.ndarray, model: BaseEstimator, threshold: float) -> Tuple[bool, float]:
    '''Score embedding using model and return prediction and propensity'''
    try:
        propensity = model.predict_proba([embedding])[0][1]
        prediction = propensity >= threshold
        return prediction, float(propensity)
    except Exception as e:
        raise RuntimeError(f"Classification failed: {str(e)}")


def convert_text_to_vectors(data, filename=None):
    """
    Modified version of your training conversion function for inference
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    data = data.replace(np.nan, '', regex=False)  # Remove regex for Series input
    return model.encode(data)

if __name__ == '__main__': # pragma: no cover
    if len(sys.argv) != 4:
        print("Error: Missing arguments!")
        print("Usage: python score.py '<text>' <model_path.joblib> <threshold>")
        print("Example: python score.py 'free viagra' model.joblib 0.5")
        sys.exit(1)

    try:
        text = sys.argv[1]
        model_path = sys.argv[2]
        threshold = float(sys.argv[3])

        model = joblib.load(model_path)
        prediction, propensity = score(text, model, threshold)

        print(f'Text: {text}')
        print(f'Model: {model_path}')
        print(f'Threshold: {threshold}')
        print(f'Prediction: {prediction}')
        print(f'Propensity: {propensity}')

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)