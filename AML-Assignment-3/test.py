import joblib
import numpy as np
import sys
import os
import requests
import time
import unittest
import subprocess
from unittest.mock import patch
from sentence_transformers import SentenceTransformer

from app import app
from score import score

sent = "This is a test text"

class TestScoreFunction(unittest.TestCase):
    """Test cases for the score function."""

    def setUp(self):
        """Set up the test environment."""
        model_path = "/Users/arkaroy/Downloads/Support_Vector_Machine_prob_final.joblib"
        self.model = joblib.load(model_path)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = 0.5
        self.test_vector = self.sentence_model.encode([sent])[0]

    def test_smoke(self):
        """Check if score function returns values properly."""
        label, prop = score(self.test_vector, self.model, self.threshold)
        self.assertIsNotNone(label)
        self.assertIsNotNone(prop)

    def test_format(self):
        """Check if the type of data meets certain requirements."""
        label, prop = score(self.test_vector, self.model, self.threshold)
        self.assertIsInstance(sent, str)
        self.assertIsInstance(self.threshold, float)
        self.assertIsInstance(label, (bool, np.bool_))
        self.assertIsInstance(prop, float)

    def test_pred_value(self):
        """Check if the label value is valid."""
        label, prop = score(self.test_vector, self.model, self.threshold)
        self.assertIn(label, [True, False])

    def test_propensity_value(self):
        """Check if propensity lies in [0,1]."""
        _, prop = score(self.test_vector, self.model, self.threshold)
        self.assertGreaterEqual(prop, 0)
        self.assertLessEqual(prop, 1)

    def test_prop_test_0(self):
        """If threshold is 0, prediction becomes True."""
        label, _ = score(self.test_vector, self.model, 0)
        self.assertTrue(label)

    def test_prop_test_1(self):
        """If threshold is 1, prediction becomes False."""
        label, _ = score(self.test_vector, self.model, 1)
        self.assertFalse(label)

    def test_spam(self):
        """Test obvious spam."""
        spam_text = self.sentence_model.encode(
            ["You have won a million dollars. Click on this link to redeem it."]
        )[0]
        label, prop = score(spam_text, self.model, self.threshold)
        self.assertTrue(label)

    def test_ham(self):
        """Test obvious ham."""
        ham_text = self.sentence_model.encode(["Dogs are cute."])[0]
        label, prop = score(ham_text, self.model, self.threshold)
        self.assertFalse(label)

class TestFlaskApp(unittest.TestCase):
    """Test cases for the Flask application."""

    def setUp(self):
        self.port = 5001
        self.url = f'http://127.0.0.1:{self.port}'
        self.process = None
        self.startup_timeout = 30

    def tearDown(self):
        if self.process:
            self.process.kill()
            time.sleep(1)

    def test_flask_basic_functionality(self):
        """Test basic server startup and shutdown."""
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-u', 'app.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0
            )

            server_ready = False
            start_time = time.time()
            output = []

            while time.time() - start_time < self.startup_timeout:
                line = self.process.stdout.readline()
                if line:
                    output.append(line.strip())
                    print(f"[SERVER] {line.strip()}")
                    if "Running on http://" in line:
                        server_ready = True
                        break
                time.sleep(0.5)

            if not server_ready:
                self.fail(f"Server startup failed. Last logs:\n" + "\n".join(output[-5:]))

            # Test home page response
            response = requests.get(self.url)
            self.assertEqual(response.status_code, 200)
            self.assertIsInstance(response.text, str)

        finally:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                time.sleep(1)
                self.assertFalse(self.is_server_running())

    def test_prediction_endpoint(self):
        """Test prediction endpoint functionality."""
        with app.test_client() as client:
            # Test valid prediction
            test_data = {'sent': 'Free prize!'}
            response = client.post('/spam', data=test_data)
            self.assertEqual(response.status_code, 200)

            # Test missing 'sent' field
            response = client.post('/spam', data={})
            self.assertEqual(response.status_code, 500)
            #self.assertIn('Missing &#39;sent&#39; field', response.text)

    def test_model_loading_failure(self):
        """Test model loading failure scenario."""
        with patch('app.MODEL_PATH', 'invalid_path.joblib'), \
             self.assertRaises(Exception):
            from importlib import reload
            reload(app)

    def test_server_error_handling(self):
        """Test server error handling."""
        with app.test_client() as client:
            with patch('app.SENTENCE_MODEL.encode') as mock_encode:
                mock_encode.side_effect = Exception("Test error")
                response = client.post('/spam', data={'sent': 'test'})
                self.assertEqual(response.status_code, 500)
                #self.assertIn('Test error', response.text)

    def test_main_execution(self):
        """Test command line execution."""
        try:
            process = subprocess.Popen(
                [sys.executable, 'app.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            time.sleep(2)
            process.terminate()
            process.wait(timeout=5)
            output, _ = process.communicate()
            #self.assertIn("Starting Flask server", output)
        except subprocess.TimeoutExpired:
            process.kill()
            self.fail("Server failed to terminate")

    def is_server_running(self):
        try:
            requests.get(self.url, timeout=1)
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False

if __name__ == '__main__': # pragma no cover
    unittest.main()