"""
Performance benchmark tests for the Nuclear Fusion Analyzer.

These tests measure the performance characteristics of various components
and provide benchmarks for optimization and monitoring purposes.
"""

import pytest
import time
import numpy as np
import pandas as pd
import psutil
import threading
from typing import Dict, List, Tuple
from unittest.mock import patch
from dataclasses import dataclass

try:
    import pytest_benchmark
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

from src.data.generator import FusionDataGenerator
from src.data.processor import FusionDataProcessor
from src.models.fusion_predictor import FusionPredictor
from src.models.anomaly_detector import FusionAnomalyDetector
from src.utils.hyperparameter_optimizer import HyperparameterOptimizer
from src.utils.performance_monitor import PerformanceMonitor


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    operation: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    metadata: Dict


class PerformanceBenchmarkSuite:
    """Performance benchmark suite for fusion analyzer components."""
    
    @pytest.fixture(scope="class")
    def benchmark_data_small(self):
        """Small dataset for quick benchmarks."""
        generator = FusionDataGenerator(random_state=42)
        return generator.generate_dataset(n_samples=1000)
    
    @pytest.fixture(scope="class")
    def benchmark_data_medium(self):
        """Medium dataset for moderate benchmarks."""
        generator = FusionDataGenerator(random_state=42)
        return generator.generate_dataset(n_samples=10000)
    
    @pytest.fixture(scope="class")
    def benchmark_data_large(self):
        """Large dataset for stress testing."""
        generator = FusionDataGenerator(random_state=42)
        return generator.generate_dataset(n_samples=50000)
    
    @pytest.fixture(scope="class")
    def trained_model(self, benchmark_data_small):
        """Pre-trained model for prediction benchmarks."""
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(benchmark_data_small)
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        predictor = FusionPredictor()
        model = predictor.train_model(X, y, model_type='random_forest')
        return predictor, X, y
    
    def measure_resource_usage(self, func, *args, **kwargs) -> Tuple[any, BenchmarkResult]:
        """Measure resource usage of a function execution."""
        process = psutil.Process()
        
        # Initial measurements
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Final measurements
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = process.cpu_percent()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = max(end_cpu - start_cpu, 0)
        
        benchmark_result = BenchmarkResult(
            operation=func.__name__,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=0,  # To be calculated based on operation
            metadata={}
        )
        
        return result, benchmark_result


class TestDataGenerationPerformance(PerformanceBenchmarkSuite):
    """Benchmark data generation performance."""
    
    def test_data_generation_small(self, benchmark):
        """Benchmark small dataset generation."""
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not available")
        
        generator = FusionDataGenerator(random_state=42)
        
        def generate_small():
            return generator.generate_dataset(n_samples=1000)
        
        result = benchmark(generate_small)
        assert len(result) == 1000
    
    def test_data_generation_medium(self, benchmark):
        """Benchmark medium dataset generation."""
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not available")
        
        generator = FusionDataGenerator(random_state=42)
        
        def generate_medium():
            return generator.generate_dataset(n_samples=10000)
        
        result = benchmark(generate_medium)
        assert len(result) == 10000
    
    def test_data_generation_scalability(self):
        """Test data generation scalability across different sizes."""
        generator = FusionDataGenerator(random_state=42)
        sizes = [1000, 5000, 10000, 25000]
        results = []
        
        for size in sizes:
            _, benchmark_result = self.measure_resource_usage(
                generator.generate_dataset,
                n_samples=size
            )
            benchmark_result.throughput = size / benchmark_result.execution_time
            benchmark_result.metadata['dataset_size'] = size
            results.append(benchmark_result)
        
        # Verify reasonable scalability
        for i in range(1, len(results)):
            prev_result = results[i-1]
            curr_result = results[i]
            
            # Execution time should scale reasonably (not exponentially)
            time_ratio = curr_result.execution_time / prev_result.execution_time
            size_ratio = curr_result.metadata['dataset_size'] / prev_result.metadata['dataset_size']
            
            assert time_ratio <= size_ratio * 2  # Allow some overhead


class TestDataProcessingPerformance(PerformanceBenchmarkSuite):
    """Benchmark data processing performance."""
    
    def test_preprocessing_small(self, benchmark, benchmark_data_small):
        """Benchmark preprocessing of small dataset."""
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not available")
        
        processor = FusionDataProcessor()
        
        def preprocess_data():
            return processor.preprocess(benchmark_data_small.copy())
        
        result = benchmark(preprocess_data)
        assert len(result) <= len(benchmark_data_small)
    
    def test_preprocessing_medium(self, benchmark, benchmark_data_medium):
        """Benchmark preprocessing of medium dataset."""
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not available")
        
        processor = FusionDataProcessor()
        
        def preprocess_data():
            return processor.preprocess(benchmark_data_medium.copy())
        
        result = benchmark(preprocess_data)
        assert len(result) <= len(benchmark_data_medium)
    
    def test_feature_engineering_performance(self, benchmark_data_medium):
        """Test feature engineering performance."""
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(benchmark_data_medium.copy())
        
        _, benchmark_result = self.measure_resource_usage(
            processor.prepare_features,
            processed_data,
            target_column='q_factor'
        )
        
        # Should complete within reasonable time
        assert benchmark_result.execution_time < 10.0  # 10 seconds
        
        # Calculate throughput
        benchmark_result.throughput = len(processed_data) / benchmark_result.execution_time
        assert benchmark_result.throughput > 500  # At least 500 samples per second


class TestModelTrainingPerformance(PerformanceBenchmarkSuite):
    """Benchmark model training performance."""
    
    def test_random_forest_training(self, benchmark, benchmark_data_small):
        """Benchmark Random Forest training."""
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not available")
        
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(benchmark_data_small.copy())
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        predictor = FusionPredictor()
        
        def train_rf():
            return predictor.train_model(X, y, model_type='random_forest')
        
        model = benchmark(train_rf)
        assert model is not None
    
    def test_gradient_boosting_training(self, benchmark, benchmark_data_small):
        """Benchmark Gradient Boosting training."""
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not available")
        
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(benchmark_data_small.copy())
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        predictor = FusionPredictor()
        
        def train_gb():
            return predictor.train_model(X, y, model_type='gradient_boosting')
        
        model = benchmark(train_gb)
        assert model is not None
    
    def test_neural_network_training(self, benchmark_data_small):
        """Test neural network training performance."""
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(benchmark_data_small.copy())
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        predictor = FusionPredictor()
        
        _, benchmark_result = self.measure_resource_usage(
            predictor.train_model,
            X, y,
            model_type='neural_network'
        )
        
        # Neural network training should complete within reasonable time
        assert benchmark_result.execution_time < 60.0  # 1 minute
    
    def test_training_scalability(self):
        """Test model training scalability with different dataset sizes."""
        sizes = [500, 1000, 2000, 4000]
        results = []
        
        for size in sizes:
            generator = FusionDataGenerator(random_state=42)
            data = generator.generate_dataset(n_samples=size)
            
            processor = FusionDataProcessor()
            processed_data = processor.preprocess(data)
            X, y = processor.prepare_features(processed_data, target_column='q_factor')
            
            predictor = FusionPredictor()
            
            _, benchmark_result = self.measure_resource_usage(
                predictor.train_model,
                X, y,
                model_type='random_forest'
            )
            
            benchmark_result.throughput = len(X) / benchmark_result.execution_time
            benchmark_result.metadata['dataset_size'] = len(X)
            results.append(benchmark_result)
        
        # Verify reasonable scalability
        for result in results:
            assert result.execution_time < 30.0  # Should complete within 30 seconds


class TestPredictionPerformance(PerformanceBenchmarkSuite):
    """Benchmark prediction performance."""
    
    def test_single_prediction_latency(self, trained_model):
        """Test single prediction latency."""
        predictor, X, y = trained_model
        single_sample = X[:1]
        
        # Measure multiple predictions to get average latency
        latencies = []
        for _ in range(100):
            start_time = time.time()
            prediction = predictor.predict(single_sample)
            end_time = time.time()
            latencies.append(end_time - start_time)
            assert len(prediction) == 1
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Latency requirements
        assert avg_latency < 0.01  # 10ms average
        assert p95_latency < 0.05  # 50ms p95
    
    def test_batch_prediction_throughput(self, trained_model):
        """Test batch prediction throughput."""
        predictor, X, y = trained_model
        
        batch_sizes = [1, 10, 100, 1000]
        throughputs = []
        
        for batch_size in batch_sizes:
            batch_data = X[:batch_size]
            
            start_time = time.time()
            predictions = predictor.predict(batch_data)
            end_time = time.time()
            
            execution_time = end_time - start_time
            throughput = batch_size / execution_time
            throughputs.append(throughput)
            
            assert len(predictions) == batch_size
        
        # Throughput should generally increase with batch size
        assert throughputs[-1] > throughputs[0]  # Largest batch should be most efficient
    
    def test_concurrent_predictions(self, trained_model):
        """Test concurrent prediction performance."""
        predictor, X, y = trained_model
        test_data = X[:100]
        
        results = []
        errors = []
        
        def make_predictions():
            try:
                start_time = time.time()
                predictions = predictor.predict(test_data)
                end_time = time.time()
                results.append(end_time - start_time)
                assert len(predictions) == len(test_data)
            except Exception as e:
                errors.append(e)
        
        # Start multiple concurrent threads
        threads = []
        num_threads = 5
        
        for _ in range(num_threads):
            thread = threading.Thread(target=make_predictions)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all threads completed successfully
        assert len(errors) == 0
        assert len(results) == num_threads
        
        # Average time per thread should be reasonable
        avg_time = np.mean(results)
        assert avg_time < 5.0  # 5 seconds


class TestAnomalyDetectionPerformance(PerformanceBenchmarkSuite):
    """Benchmark anomaly detection performance."""
    
    def test_anomaly_detection_training(self, benchmark, benchmark_data_small):
        """Benchmark anomaly detection model training."""
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not available")
        
        detector = FusionAnomalyDetector()
        training_data = benchmark_data_small.drop(columns=['timestamp'] if 'timestamp' in benchmark_data_small.columns else [])
        
        def train_detector():
            return detector.fit(training_data)
        
        result = benchmark(train_detector)
        assert result is not None
    
    def test_anomaly_detection_inference(self, benchmark_data_small):
        """Test anomaly detection inference performance."""
        detector = FusionAnomalyDetector()
        training_data = benchmark_data_small.drop(columns=['timestamp'] if 'timestamp' in benchmark_data_small.columns else [])
        detector.fit(training_data)
        
        test_data = training_data[:100]
        
        _, benchmark_result = self.measure_resource_usage(
            detector.detect_anomalies,
            test_data
        )
        
        benchmark_result.throughput = len(test_data) / benchmark_result.execution_time
        
        # Should achieve good throughput
        assert benchmark_result.throughput > 1000  # At least 1000 samples per second
        assert benchmark_result.execution_time < 1.0  # Complete within 1 second


class TestHyperparameterOptimizationPerformance(PerformanceBenchmarkSuite):
    """Benchmark hyperparameter optimization performance."""
    
    def test_grid_search_performance(self, benchmark_data_small):
        """Test grid search optimization performance."""
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(benchmark_data_small.copy())
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        optimizer = HyperparameterOptimizer()
        
        _, benchmark_result = self.measure_resource_usage(
            optimizer.optimize_model,
            X, y,
            model_type='random_forest',
            strategy='grid_search',
            n_trials=9  # 3x3 grid
        )
        
        # Should complete within reasonable time
        assert benchmark_result.execution_time < 120.0  # 2 minutes
    
    def test_random_search_performance(self, benchmark_data_small):
        """Test random search optimization performance."""
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(benchmark_data_small.copy())
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        optimizer = HyperparameterOptimizer()
        
        _, benchmark_result = self.measure_resource_usage(
            optimizer.optimize_model,
            X, y,
            model_type='random_forest',
            strategy='random_search',
            n_trials=10
        )
        
        # Should complete within reasonable time
        assert benchmark_result.execution_time < 90.0  # 1.5 minutes


class TestMemoryUsagePatterns(PerformanceBenchmarkSuite):
    """Test memory usage patterns and potential leaks."""
    
    def test_memory_usage_during_training(self):
        """Monitor memory usage during model training."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_samples = []
        
        def monitor_memory():
            for _ in range(60):  # Monitor for 60 seconds
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                time.sleep(1)
        
        # Start memory monitoring
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Perform training
        generator = FusionDataGenerator(random_state=42)
        data = generator.generate_dataset(n_samples=5000)
        
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(data)
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        
        predictor = FusionPredictor()
        model = predictor.train_model(X, y, model_type='random_forest')
        
        # Wait a bit for memory monitoring
        time.sleep(5)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        # Memory usage should be reasonable
        memory_increase = final_memory - initial_memory
        assert memory_increase < 1000  # Less than 1GB increase
        
        # Check for memory leaks (memory should stabilize)
        if len(memory_samples) > 10:
            recent_samples = memory_samples[-10:]
            memory_variance = np.var(recent_samples)
            assert memory_variance < 100  # Low variance indicates stable memory usage
    
    def test_prediction_memory_efficiency(self, trained_model):
        """Test memory efficiency during predictions."""
        predictor, X, y = trained_model
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Make many predictions
        for _ in range(100):
            batch_data = X[:100]
            predictions = predictor.predict(batch_data)
            assert len(predictions) == 100
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal for predictions
        assert memory_increase < 100  # Less than 100MB increase


@pytest.mark.integration
class TestEndToEndPerformance(PerformanceBenchmarkSuite):
    """Test end-to-end performance of the complete system."""
    
    def test_complete_pipeline_performance(self):
        """Test performance of the complete ML pipeline."""
        total_start_time = time.time()
        
        # Step 1: Data generation
        step_start = time.time()
        generator = FusionDataGenerator(random_state=42)
        data = generator.generate_dataset(n_samples=5000)
        generation_time = time.time() - step_start
        
        # Step 2: Data processing
        step_start = time.time()
        processor = FusionDataProcessor()
        processed_data = processor.preprocess(data)
        X, y = processor.prepare_features(processed_data, target_column='q_factor')
        processing_time = time.time() - step_start
        
        # Step 3: Model training
        step_start = time.time()
        predictor = FusionPredictor()
        model = predictor.train_model(X, y, model_type='random_forest')
        training_time = time.time() - step_start
        
        # Step 4: Model evaluation
        step_start = time.time()
        metrics = predictor.evaluate_model(X, y)
        evaluation_time = time.time() - step_start
        
        # Step 5: Predictions
        step_start = time.time()
        predictions = predictor.predict(X[:1000])
        prediction_time = time.time() - step_start
        
        total_time = time.time() - total_start_time
        
        # Performance assertions
        assert generation_time < 10.0  # Data generation
        assert processing_time < 15.0  # Data processing
        assert training_time < 60.0   # Model training
        assert evaluation_time < 10.0  # Model evaluation
        assert prediction_time < 5.0   # Predictions
        assert total_time < 100.0      # Total pipeline
        
        # Verify quality
        assert metrics['r2'] > 0.5  # Reasonable model performance
        assert len(predictions) == 1000
        assert not np.isnan(predictions).any()
    
    def test_api_endpoint_performance(self):
        """Test API endpoint performance (if available)."""
        try:
            from api.app import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            
            # Test single prediction latency
            test_data = {
                "magnetic_field": 5.3,
                "plasma_current": 15.0,
                "electron_density": 1.0e20,
                "ion_temperature": 20.0,
                "electron_temperature": 15.0,
                "neutral_beam_power": 50.0,
                "rf_heating_power": 30.0
            }
            
            latencies = []
            for _ in range(10):
                start_time = time.time()
                response = client.post("/predict", json=test_data)
                end_time = time.time()
                
                assert response.status_code == 200
                latencies.append(end_time - start_time)
            
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            # API latency requirements
            assert avg_latency < 1.0  # 1 second average
            assert p95_latency < 2.0  # 2 seconds p95
            
        except ImportError:
            pytest.skip("API components not available for performance testing")