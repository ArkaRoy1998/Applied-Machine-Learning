import score
import joblib
import numpy as np
import os
import requests
import time
import unittest
from multiprocessing import Process
from app import app

# Load the SVM model
filename = "Support_Vector_Machine_prob_final.joblib"

svm_model = joblib.load(filename)

# Example sentence for testing
sentence = "I enjoy outdoor activities."

# Threshold for classification
threshold = 0.5

# Score the example sentence
example_label, example_propensity = score.score(sentence, svm_model, threshold)
class TestFunction(unittest.TestCase):
    def test_smoke(self):
        #Check if score function returns values properly.
        self.assertIsNotNone(example_label)
        self.assertIsNotNone(example_propensity)

    def test_format(self):
        #Check if the type of data meets certain requirements.
        self.assertIsInstance(sentence, str)
        self.assertIsInstance(threshold, float)
        self.assertIsInstance(example_label, int)
        self.assertIsInstance(example_propensity, float
                              )

    def test_pred_value(self):
        #Check if the label value is in {0,1}
        self.assertIn(example_label, [0, 1])

    def test_propensity_value(self):
        #Check if propensity lies in [0,1].
        self.assertGreaterEqual(example_propensity, 0)
        self.assertLessEqual(example_propensity, 1)

    def test_prop_test_0(self):
        """If threshold is 0, prediction should always be 1."""
        label, prop = score.score("Win $1000 now! Click here!", svm_model, 0)
        self.assertEqual(label, 1, msg=f"Expected 1 since threshold=0, got {label} (propensity={prop})")

    def test_prop_test_1(self):
        """If threshold is 1, prediction should always be 0."""
        label, prop = score.score("Win $1000 now! Click here!", svm_model, 1)
        self.assertEqual(label, 0, msg=f"Expected 0 since threshold=1, got {label} (propensity={prop})")

    def test_spam(self):
        #Test with an obvious spam sentence.
        label, prop = score.score("You have won a prize. Click here to claim it.", svm_model, threshold)
        self.assertEqual(label, 1)

    def test_ham(self):
        #Test with an obvious ham sentence.
        label, prop = score.score("The meeting has been rescheduled to next week.", svm_model, threshold)
        self.assertEqual(label, 0)

@unittest.skip("Skipping Flask tests for coverage")
class TestFlask(unittest.TestCase):
    #catTest cases for the Flask application.


    def test_flask(self):
        """Test the Flask application."""
        # Launch the Flask app using a subprocess
        os.system('python app.py &')

        # Wait for the app to start up
        time.sleep(1)

        # Make a request to the endpoint
        response = requests.get('http://127.0.0.1:5000/')

        # Assert that the response is what we expect
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.text, str)

        # Shut down the Flask app
        os.system('kill $(lsof -t -i:5000)')


@unittest.skip("Skip Docker test during coverage collection")
class TestDocker(unittest.TestCase):
    def test_docker(self):
        import subprocess

        # Build Docker image
        subprocess.run(["docker", "build", "-t", "flask-sentiment-app", "."], check=True)

        # Run container in background
        container = subprocess.Popen(
            ["docker", "run", "-p", "5003:5000", "--rm", "--name", "test_container", "flask-sentiment-app"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        time.sleep(10)  # Wait for Flask to start

        try:
            # Send request to /score endpoint
            response = requests.post("http://localhost:5003/score", json={"sentence": "I love this!"})
            self.assertEqual(response.status_code, 200)
            self.assertIn("result", response.json())
        finally:
            # Stop the container
            subprocess.run(["docker", "stop", "test_container"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)




if __name__ == '__main__': # pragma no cover
    unittest.main()
