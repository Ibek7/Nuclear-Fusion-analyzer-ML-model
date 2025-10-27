"""
API Testing and Validation Framework.

This module provides:
- Automated API endpoint testing
- Request/response validation
- Performance testing and benchmarking
- Load testing capabilities
- API contract testing
- Mock API server for testing
"""

import asyncio
import json
import time
import statistics
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import random

# HTTP client imports
try:
    import requests
    import aiohttp
    HAS_HTTP_CLIENTS = True
except ImportError:
    HAS_HTTP_CLIENTS = False
    requests = None
    aiohttp = None

logger = logging.getLogger(__name__)


@dataclass
class TestRequest:
    """Represents a test request."""
    method: str
    url: str
    headers: Optional[Dict[str, str]] = None
    data: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, str]] = None
    timeout: float = 30.0
    expected_status: int = 200
    
    def __post_init__(self):
        """Validate test request."""
        if self.headers is None:
            self.headers = {}
        if self.method.upper() not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
            raise ValueError(f"Unsupported HTTP method: {self.method}")


@dataclass
class TestResponse:
    """Represents a test response."""
    status_code: int
    response_time: float
    data: Dict[str, Any]
    headers: Dict[str, str]
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestResult:
    """Represents test execution result."""
    test_name: str
    request: TestRequest
    response: TestResponse
    passed: bool
    error_message: Optional[str] = None
    assertions_passed: int = 0
    assertions_failed: int = 0
    execution_time: float = 0.0


@dataclass
class PerformanceMetrics:
    """Performance testing metrics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    min_response_time: float
    max_response_time: float
    median_response_time: float
    percentile_95: float
    percentile_99: float
    requests_per_second: float
    error_rate: float
    
    @classmethod
    def from_response_times(cls, response_times: List[float], total_time: float) -> 'PerformanceMetrics':
        """Create metrics from response times."""
        if not response_times:
            return cls(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_response_time=0.0,
                min_response_time=0.0,
                max_response_time=0.0,
                median_response_time=0.0,
                percentile_95=0.0,
                percentile_99=0.0,
                requests_per_second=0.0,
                error_rate=0.0
            )
        
        sorted_times = sorted(response_times)
        
        return cls(
            total_requests=len(response_times),
            successful_requests=len(response_times),
            failed_requests=0,
            average_response_time=statistics.mean(response_times),
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            median_response_time=statistics.median(response_times),
            percentile_95=sorted_times[int(len(sorted_times) * 0.95)],
            percentile_99=sorted_times[int(len(sorted_times) * 0.99)],
            requests_per_second=len(response_times) / total_time if total_time > 0 else 0,
            error_rate=0.0
        )


class APIValidator:
    """Validates API requests and responses."""
    
    def __init__(self):
        """Initialize API validator."""
        self.validation_rules = {}
        logger.info("APIValidator initialized")
    
    def add_validation_rule(self, endpoint: str, rule: Callable[[TestResponse], bool]):
        """
        Add validation rule for endpoint.
        
        Args:
            endpoint: API endpoint pattern.
            rule: Validation function.
        """
        if endpoint not in self.validation_rules:
            self.validation_rules[endpoint] = []
        self.validation_rules[endpoint].append(rule)
    
    def validate_response(self, endpoint: str, response: TestResponse) -> List[str]:
        """
        Validate response against rules.
        
        Args:
            endpoint: API endpoint.
            response: Response to validate.
            
        Returns:
            List of validation errors.
        """
        errors = []
        
        # Get matching rules
        rules = self.validation_rules.get(endpoint, [])
        
        for rule in rules:
            try:
                if not rule(response):
                    errors.append(f"Validation rule failed for endpoint {endpoint}")
            except Exception as e:
                errors.append(f"Validation rule error: {e}")
        
        return errors
    
    def validate_fusion_metrics_response(self, response: TestResponse) -> bool:
        """Validate fusion metrics response."""
        try:
            data = response.data
            
            # Check required fields
            required_fields = ['triple_product', 'beta', 'ignition_conditions']
            for field in required_fields:
                if field not in data:
                    return False
            
            # Check data types and ranges
            if not isinstance(data['triple_product'], (int, float)) or data['triple_product'] < 0:
                return False
            
            if not isinstance(data['beta'], (int, float)) or data['beta'] < 0 or data['beta'] > 1:
                return False
            
            if not isinstance(data['ignition_conditions'], bool):
                return False
            
            # Optional fields validation
            if 'fusion_power' in data and data['fusion_power'] is not None:
                if not isinstance(data['fusion_power'], (int, float)) or data['fusion_power'] < 0:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def validate_ml_prediction_response(self, response: TestResponse) -> bool:
        """Validate ML prediction response."""
        try:
            data = response.data
            
            # Check required fields
            required_fields = ['prediction', 'confidence', 'model_version']
            for field in required_fields:
                if field not in data:
                    return False
            
            # Check confidence range
            if not isinstance(data['confidence'], (int, float)) or not (0 <= data['confidence'] <= 1):
                return False
            
            # Check model version format
            if not isinstance(data['model_version'], str) or not data['model_version']:
                return False
            
            return True
            
        except Exception:
            return False


class APITester:
    """Automated API testing framework."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize API tester.
        
        Args:
            base_url: Base URL of the API.
            api_key: Optional API key for authentication.
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.validator = APIValidator()
        self.test_results = []
        
        # Setup default validation rules
        self._setup_default_validation_rules()
        
        logger.info(f"APITester initialized for {base_url}")
    
    def _setup_default_validation_rules(self):
        """Setup default validation rules."""
        self.validator.add_validation_rule(
            "/api/v1/fusion/analyze",
            self.validator.validate_fusion_metrics_response
        )
        self.validator.add_validation_rule(
            "/api/v1/ml/predict",
            self.validator.validate_ml_prediction_response
        )
    
    def _prepare_headers(self, custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Prepare request headers."""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'FusionAPI-Tester/1.0'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        if custom_headers:
            headers.update(custom_headers)
        
        return headers
    
    def execute_test(self, test_name: str, test_request: TestRequest) -> TestResult:
        """
        Execute single test.
        
        Args:
            test_name: Name of the test.
            test_request: Test request configuration.
            
        Returns:
            Test result.
        """
        if not HAS_HTTP_CLIENTS:
            return TestResult(
                test_name=test_name,
                request=test_request,
                response=TestResponse(
                    status_code=500,
                    response_time=0.0,
                    data={},
                    headers={},
                    error="HTTP client libraries not available"
                ),
                passed=False,
                error_message="HTTP client libraries not available"
            )
        
        start_time = time.time()
        
        try:
            # Prepare request
            url = f"{self.base_url}{test_request.url}"
            headers = self._prepare_headers(test_request.headers)
            
            # Make request
            if test_request.method.upper() == 'GET':
                response = requests.get(
                    url,
                    headers=headers,
                    params=test_request.params,
                    timeout=test_request.timeout
                )
            elif test_request.method.upper() == 'POST':
                response = requests.post(
                    url,
                    headers=headers,
                    json=test_request.data,
                    params=test_request.params,
                    timeout=test_request.timeout
                )
            else:
                return TestResult(
                    test_name=test_name,
                    request=test_request,
                    response=TestResponse(
                        status_code=500,
                        response_time=0.0,
                        data={},
                        headers={},
                        error=f"Unsupported method: {test_request.method}"
                    ),
                    passed=False,
                    error_message=f"Unsupported method: {test_request.method}"
                )
            
            execution_time = time.time() - start_time
            
            # Parse response
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {"text": response.text}
            
            test_response = TestResponse(
                status_code=response.status_code,
                response_time=execution_time,
                data=response_data,
                headers=dict(response.headers)
            )
            
            # Validate response
            validation_errors = self.validator.validate_response(test_request.url, test_response)
            
            # Check assertions
            passed = (
                response.status_code == test_request.expected_status and
                len(validation_errors) == 0
            )
            
            test_result = TestResult(
                test_name=test_name,
                request=test_request,
                response=test_response,
                passed=passed,
                error_message="; ".join(validation_errors) if validation_errors else None,
                assertions_passed=1 if passed else 0,
                assertions_failed=0 if passed else 1,
                execution_time=execution_time
            )
            
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            test_result = TestResult(
                test_name=test_name,
                request=test_request,
                response=TestResponse(
                    status_code=0,
                    response_time=execution_time,
                    data={},
                    headers={},
                    error=str(e)
                ),
                passed=False,
                error_message=str(e),
                execution_time=execution_time
            )
            
            self.test_results.append(test_result)
            return test_result
    
    def run_test_suite(self, test_suite: Dict[str, TestRequest]) -> Dict[str, TestResult]:
        """
        Run complete test suite.
        
        Args:
            test_suite: Dictionary of test name to test request.
            
        Returns:
            Dictionary of test results.
        """
        results = {}
        
        logger.info(f"Running test suite with {len(test_suite)} tests")
        
        for test_name, test_request in test_suite.items():
            logger.info(f"Executing test: {test_name}")
            result = self.execute_test(test_name, test_request)
            results[test_name] = result
            
            if result.passed:
                logger.info(f"✅ {test_name} passed")
            else:
                logger.error(f"❌ {test_name} failed: {result.error_message}")
        
        return results
    
    def generate_test_report(self) -> Dict[str, Any]:
        """
        Generate test report.
        
        Returns:
            Test report summary.
        """
        if not self.test_results:
            return {"message": "No tests executed"}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests
        
        total_execution_time = sum(result.execution_time for result in self.test_results)
        average_execution_time = total_execution_time / total_tests
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests) * 100,
                "total_execution_time": total_execution_time,
                "average_execution_time": average_execution_time
            },
            "test_results": [
                {
                    "test_name": result.test_name,
                    "passed": result.passed,
                    "execution_time": result.execution_time,
                    "status_code": result.response.status_code,
                    "error_message": result.error_message
                }
                for result in self.test_results
            ]
        }


class LoadTester:
    """Load testing for API endpoints."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize load tester.
        
        Args:
            base_url: Base URL of the API.
            api_key: Optional API key for authentication.
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        logger.info(f"LoadTester initialized for {base_url}")
    
    def run_load_test(self,
                     endpoint: str,
                     method: str = "GET",
                     data: Optional[Dict[str, Any]] = None,
                     concurrent_users: int = 10,
                     requests_per_user: int = 100,
                     ramp_up_time: float = 10.0) -> PerformanceMetrics:
        """
        Run load test on endpoint.
        
        Args:
            endpoint: API endpoint to test.
            method: HTTP method.
            data: Request data.
            concurrent_users: Number of concurrent users.
            requests_per_user: Requests per user.
            ramp_up_time: Time to ramp up all users (seconds).
            
        Returns:
            Performance metrics.
        """
        if not HAS_HTTP_CLIENTS:
            logger.error("HTTP client libraries not available for load testing")
            return PerformanceMetrics.from_response_times([], 0.0)
        
        logger.info(f"Starting load test: {concurrent_users} users, {requests_per_user} requests each")
        
        response_times = []
        errors = []
        start_time = time.time()
        
        def make_request(user_id: int) -> List[float]:
            """Make requests for a single user."""
            user_response_times = []
            
            # Stagger user start times
            delay = (ramp_up_time / concurrent_users) * user_id
            time.sleep(delay)
            
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            for _ in range(requests_per_user):
                request_start = time.time()
                
                try:
                    url = f"{self.base_url}{endpoint}"
                    
                    if method.upper() == 'GET':
                        response = requests.get(url, headers=headers, timeout=30)
                    elif method.upper() == 'POST':
                        response = requests.post(url, headers=headers, json=data, timeout=30)
                    else:
                        continue
                    
                    request_time = time.time() - request_start
                    
                    if response.status_code < 400:
                        user_response_times.append(request_time)
                    else:
                        errors.append(f"HTTP {response.status_code}")
                        
                except Exception as e:
                    errors.append(str(e))
                
                # Small delay between requests
                time.sleep(0.1)
            
            return user_response_times
        
        # Execute load test with thread pool
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(make_request, user_id)
                for user_id in range(concurrent_users)
            ]
            
            for future in as_completed(futures):
                try:
                    user_times = future.result()
                    response_times.extend(user_times)
                except Exception as e:
                    logger.error(f"User thread error: {e}")
                    errors.append(str(e))
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        total_requests = concurrent_users * requests_per_user
        successful_requests = len(response_times)
        failed_requests = total_requests - successful_requests
        
        if response_times:
            metrics = PerformanceMetrics.from_response_times(response_times, total_time)
            metrics.total_requests = total_requests
            metrics.failed_requests = failed_requests
            metrics.error_rate = (failed_requests / total_requests) * 100
        else:
            metrics = PerformanceMetrics(
                total_requests=total_requests,
                successful_requests=0,
                failed_requests=failed_requests,
                average_response_time=0.0,
                min_response_time=0.0,
                max_response_time=0.0,
                median_response_time=0.0,
                percentile_95=0.0,
                percentile_99=0.0,
                requests_per_second=0.0,
                error_rate=100.0
            )
        
        logger.info(f"Load test completed: {successful_requests}/{total_requests} successful")
        return metrics


def create_fusion_api_test_suite(base_url: str) -> Dict[str, TestRequest]:
    """
    Create test suite for Fusion API.
    
    Args:
        base_url: Base URL of the API.
        
    Returns:
        Test suite dictionary.
    """
    return {
        "health_check": TestRequest(
            method="GET",
            url="/health",
            expected_status=200
        ),
        "api_info": TestRequest(
            method="GET",
            url="/api/info",
            expected_status=200
        ),
        "fusion_analyze_valid": TestRequest(
            method="POST",
            url="/api/v1/fusion/analyze",
            data={
                "temperature": 100e6,
                "density": 1e20,
                "magnetic_field": 5.3,
                "confinement_time": 1.2
            },
            expected_status=200
        ),
        "fusion_analyze_minimal": TestRequest(
            method="POST",
            url="/api/v1/fusion/analyze",
            data={
                "temperature": 50e6,
                "density": 5e19
            },
            expected_status=200
        ),
        "fusion_analyze_invalid_temp": TestRequest(
            method="POST",
            url="/api/v1/fusion/analyze",
            data={
                "temperature": -100,
                "density": 1e20
            },
            expected_status=400
        ),
        "ml_predict_valid": TestRequest(
            method="POST",
            url="/api/v1/ml/predict",
            data={
                "features": {
                    "plasma_temperature": 100e6,
                    "plasma_density": 1e20,
                    "magnetic_field": 5.3
                },
                "model_name": "fusion_performance_predictor",
                "confidence_threshold": 0.8
            },
            expected_status=200
        ),
        "pipeline_status": TestRequest(
            method="GET",
            url="/api/v1/pipelines/status",
            expected_status=200
        )
    }


def create_api_tester(base_url: str, api_key: Optional[str] = None) -> APITester:
    """
    Create API tester instance.
    
    Args:
        base_url: Base URL of the API.
        api_key: Optional API key.
        
    Returns:
        API tester instance.
    """
    return APITester(base_url, api_key)


def create_load_tester(base_url: str, api_key: Optional[str] = None) -> LoadTester:
    """
    Create load tester instance.
    
    Args:
        base_url: Base URL of the API.
        api_key: Optional API key.
        
    Returns:
        Load tester instance.
    """
    return LoadTester(base_url, api_key)