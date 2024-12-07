# tests/test_config.py
import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(root_dir)
sys.path.append(root_dir)

import unittest
from src.system.config_manager import Config

class TestConfig(unittest.TestCase):

    def test_default_initialization(self):
        # Create an instance with default configurations
        config = Config()
        
        # Check default model and chunk_size
        self.assertEqual(config.get_model_config()['model_name'], "tiny.en")
        self.assertEqual(config.get_model_config()['chunk_size'], 500)
        self.assertEqual(config.get_reporting_config()['report_interval'], 10)

    def test_custom_initialization(self):
        # Create an instance with custom configurations
        config = Config(model_name="base", chunk_size=1000, report_interval=15)
        
        # Check custom values
        self.assertEqual(config.get_model_config()['model_name'], "base")
        self.assertEqual(config.get_model_config()['chunk_size'], 1000)
        self.assertEqual(config.get_reporting_config()['report_interval'], 15)

    def test_set_model(self):
        config = Config()
        
        # Set a valid model name
        config.set_model("small")
        self.assertEqual(config.get_model_config()['model_name'], "small")
        
        # Test setting an invalid model (should raise ValueError)
        with self.assertRaises(ValueError):
            config.set_model("invalid_model")

    def test_set_chunk_size(self):
        config = Config()
        
        # Set a valid chunk size
        config.set_chunk_size(2000)
        self.assertEqual(config.get_model_config()['chunk_size'], 2000)
        
        # Set a different chunk size
        config.set_chunk_size(100)
        self.assertEqual(config.get_model_config()['chunk_size'], 100)

    def test_set_report_interval(self):
        config = Config()
        
        # Set a valid report interval
        config.set_report_interval(5)
        self.assertEqual(config.get_reporting_config()['report_interval'], 5)

    def test_get_all_models(self):
        config = Config()
        
        # Get all available models
        models = config.get_all_models()
        
        # Check that the models list contains the correct models
        self.assertIn("tiny.en", models)
        self.assertIn("large", models)
        self.assertIn("medium.en", models)
        self.assertIn("turbo", models)

if __name__ == "__main__":
    unittest.main()
