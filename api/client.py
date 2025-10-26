"""
Example client for the Nuclear Fusion Analyzer API.

Demonstrates how to interact with the REST API endpoints.
"""

import requests
import json
import time
from typing import Dict, Any, List


class FusionAnalyzerClient:
    """Client for Nuclear Fusion Analyzer API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def train_model(self, model_name: str, n_samples: int = 10000, 
                   hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train a model.
        
        Args:
            model_name: Name of model to train
            n_samples: Number of training samples
            hyperparameters: Custom hyperparameters
        """
        payload = {
            "model_name": model_name,
            "n_samples": n_samples
        }
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters
        
        response = self.session.post(f"{self.base_url}/models/train", json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict(self, plasma_params: Dict[str, float], 
                heating_params: Dict[str, float] = None,
                fuel_params: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Make a fusion prediction.
        
        Args:
            plasma_params: Plasma parameters
            heating_params: Heating system parameters
            fuel_params: Fuel system parameters
        """
        payload = {"plasma": plasma_params}
        if heating_params:
            payload["heating"] = heating_params
        if fuel_params:
            payload["fuel"] = fuel_params
        
        response = self.session.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make batch predictions.
        
        Args:
            predictions: List of prediction requests
        """
        payload = {"predictions": predictions}
        response = self.session.post(f"{self.base_url}/predict/batch", json=payload)
        response.raise_for_status()
        return response.json()
    
    def detect_anomaly(self, data: Dict[str, float], 
                      detectors: List[str] = None) -> Dict[str, Any]:
        """
        Detect anomalies in data.
        
        Args:
            data: Fusion parameters to check
            detectors: Specific detectors to use
        """
        payload = {"data": data}
        if detectors:
            payload["detectors"] = detectors
        
        response = self.session.post(f"{self.base_url}/anomaly/detect", json=payload)
        response.raise_for_status()
        return response.json()
    
    def generate_data(self, n_samples: int = 1000, anomaly_rate: float = 0.05,
                     include_time_series: bool = False) -> Dict[str, Any]:
        """
        Generate synthetic data.
        
        Args:
            n_samples: Number of samples to generate
            anomaly_rate: Fraction of anomalous samples
            include_time_series: Whether to include time series
        """
        payload = {
            "n_samples": n_samples,
            "anomaly_rate": anomaly_rate,
            "include_time_series": include_time_series
        }
        
        response = self.session.post(f"{self.base_url}/data/generate", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        response = self.session.get(f"{self.base_url}/config")
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the client."""
    # Initialize client
    client = FusionAnalyzerClient()
    
    print("Nuclear Fusion Analyzer API Client Example")
    print("=" * 50)
    
    try:
        # Check health
        print("\n1. Health Check:")
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Uptime: {health['uptime']:.2f} seconds")
        
        # List models
        print("\n2. Available Models:")
        models = client.list_models()
        for model in models['models']:
            print(f"  - {model['name']} ({model['type']}): {'Trained' if model['trained'] else 'Not trained'}")
        
        # Train a model if none are trained
        trained_models = [m for m in models['models'] if m['trained']]
        if not trained_models:
            print("\n3. Training Random Forest Model:")
            training_result = client.train_model('random_forest', n_samples=1000)
            print(f"Training successful: {training_result['success']}")
            print(f"Training time: {training_result['training_time']:.2f} seconds")
            print(f"Validation R²: {training_result['performance_metrics']['val_r2']:.4f}")
        else:
            print(f"\n3. Using existing trained model: {trained_models[0]['name']}")
        
        # Make a prediction
        print("\n4. Making Prediction:")
        plasma_params = {
            "magnetic_field": 5.3,
            "plasma_current": 15.0,
            "electron_density": 1.0e20,
            "ion_temperature": 20.0,
            "electron_temperature": 15.0
        }
        
        heating_params = {
            "neutral_beam_power": 50.0,
            "rf_heating_power": 30.0,
            "ohmic_heating_power": 10.0,
            "heating_efficiency": 0.85
        }
        
        prediction = client.predict(plasma_params, heating_params)
        print(f"Predicted Q factor: {prediction['q_factor']:.3f}")
        print(f"Confidence: {prediction['confidence']:.3f}")
        print(f"Performance category: {prediction['performance_category']}")
        
        # Detect anomalies
        print("\n5. Anomaly Detection:")
        test_data = {
            "magnetic_field": 5.3,
            "plasma_current": 15.0,
            "electron_density": 1.0e20,
            "ion_temperature": 20.0,
            "q_factor": 1.2
        }
        
        anomaly_result = client.detect_anomaly(test_data)
        print(f"Is anomaly: {anomaly_result['is_anomaly']}")
        print(f"Anomaly score: {anomaly_result['anomaly_score']:.3f}")
        if anomaly_result['anomaly_type']:
            print(f"Anomaly type: {anomaly_result['anomaly_type']}")
        
        # Generate data
        print("\n6. Generating Synthetic Data:")
        data_result = client.generate_data(n_samples=100, anomaly_rate=0.1)
        print(f"Generated {data_result['n_samples']} samples")
        print(f"Columns: {len(data_result['columns'])}")
        print(f"Mean Q factor: {data_result['statistics']['mean_q_factor']:.3f}")
        print(f"Breakeven fraction: {data_result['statistics']['breakeven_fraction']:.3f}")
        
        print("\n✅ All API endpoints working correctly!")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()