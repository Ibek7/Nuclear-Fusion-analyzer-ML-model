"""
Real-time Stream Processing for Nuclear Fusion Data.

This module provides:
- Apache Kafka integration for data streaming
- Real-time data processing with Apache Flink/Spark Streaming
- Stream analytics and pattern detection
- Event-driven architecture for fusion systems
- Real-time alerts and notifications
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from datetime import datetime, timedelta
import logging
import threading
import queue
from abc import ABC, abstractmethod

# Kafka imports (optional)
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False
    KafkaProducer = None
    KafkaConsumer = None

# Asyncio imports for websockets (optional)
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

import pandas as pd
import numpy as np
from collections import deque, defaultdict
import time

logger = logging.getLogger(__name__)


class StreamMessage:
    """Represents a message in the data stream."""
    
    def __init__(self, 
                 data: Dict[str, Any], 
                 timestamp: Optional[datetime] = None,
                 source: Optional[str] = None,
                 message_id: Optional[str] = None):
        """
        Initialize stream message.
        
        Args:
            data: Message payload.
            timestamp: Message timestamp.
            source: Message source identifier.
            message_id: Unique message identifier.
        """
        self.data = data
        self.timestamp = timestamp or datetime.now()
        self.source = source
        self.message_id = message_id or f"{int(time.time() * 1000)}"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "message_id": self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamMessage':
        """Create message from dictionary."""
        return cls(
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            source=data.get("source"),
            message_id=data.get("message_id")
        )


class StreamProcessor(ABC):
    """Abstract base class for stream processors."""
    
    @abstractmethod
    async def process_message(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Process a single message."""
        pass
    
    @abstractmethod
    async def process_batch(self, messages: List[StreamMessage]) -> List[StreamMessage]:
        """Process a batch of messages."""
        pass


class FusionDataStreamProcessor(StreamProcessor):
    """Stream processor for nuclear fusion data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fusion data stream processor.
        
        Args:
            config: Processor configuration.
        """
        self.config = config or {}
        self.window_size = self.config.get("window_size", 100)  # Number of messages in sliding window
        self.alert_thresholds = self.config.get("alert_thresholds", {})
        
        # Sliding window for temporal analysis
        self.data_window = deque(maxlen=self.window_size)
        self.metrics_history = deque(maxlen=1000)  # Keep more history for trends
        
        # Alert state
        self.last_alert_time = {}
        self.alert_cooldown = timedelta(minutes=5)  # Prevent alert spam
        
        logger.info(f"FusionDataStreamProcessor initialized with window size: {self.window_size}")
    
    async def process_message(self, message: StreamMessage) -> Optional[StreamMessage]:
        """
        Process a single fusion data message.
        
        Args:
            message: Input message.
            
        Returns:
            Processed message or None.
        """
        try:
            # Extract plasma parameters
            data = message.data
            
            # Validate required fields
            required_fields = ["plasma_temperature", "plasma_density"]
            if not all(field in data for field in required_fields):
                logger.warning(f"Missing required fields in message: {message.message_id}")
                return None
            
            # Add to sliding window
            self.data_window.append(message)
            
            # Calculate real-time metrics
            metrics = self._calculate_realtime_metrics()
            
            # Check for alerts
            alerts = self._check_alert_conditions(data, metrics)
            
            # Create processed message
            processed_data = {
                **data,
                "realtime_metrics": metrics,
                "alerts": alerts,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            processed_message = StreamMessage(
                data=processed_data,
                timestamp=message.timestamp,
                source=f"processed_{message.source}",
                message_id=f"proc_{message.message_id}"
            )
            
            # Store metrics history
            self.metrics_history.append({
                "timestamp": message.timestamp,
                "metrics": metrics,
                "alerts": alerts
            })
            
            return processed_message
            
        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
            return None
    
    async def process_batch(self, messages: List[StreamMessage]) -> List[StreamMessage]:
        """
        Process a batch of messages.
        
        Args:
            messages: List of input messages.
            
        Returns:
            List of processed messages.
        """
        processed_messages = []
        
        for message in messages:
            processed = await self.process_message(message)
            if processed:
                processed_messages.append(processed)
        
        # Perform batch-level analysis
        if processed_messages:
            batch_metrics = self._calculate_batch_metrics(processed_messages)
            
            # Add batch metrics to last message
            if processed_messages:
                last_message = processed_messages[-1]
                last_message.data["batch_metrics"] = batch_metrics
        
        return processed_messages
    
    def _calculate_realtime_metrics(self) -> Dict[str, float]:
        """Calculate real-time metrics from sliding window."""
        if not self.data_window:
            return {}
        
        # Extract data from window
        temperatures = []
        densities = []
        confinement_times = []
        
        for msg in self.data_window:
            data = msg.data
            if "plasma_temperature" in data:
                temperatures.append(data["plasma_temperature"])
            if "plasma_density" in data:
                densities.append(data["plasma_density"])
            if "confinement_time" in data:
                confinement_times.append(data["confinement_time"])
        
        metrics = {}
        
        # Basic statistics
        if temperatures:
            metrics["avg_temperature"] = np.mean(temperatures)
            metrics["temp_std"] = np.std(temperatures)
            metrics["temp_trend"] = self._calculate_trend(temperatures)
        
        if densities:
            metrics["avg_density"] = np.mean(densities)
            metrics["density_std"] = np.std(densities)
            metrics["density_trend"] = self._calculate_trend(densities)
        
        # Fusion-specific metrics
        if temperatures and densities:
            # Calculate triple product if confinement time available
            if confinement_times:
                triple_products = [t * d * c for t, d, c in zip(temperatures, densities, confinement_times)]
                metrics["avg_triple_product"] = np.mean(triple_products)
                metrics["max_triple_product"] = np.max(triple_products)
            
            # Calculate beta (plasma pressure / magnetic pressure)
            # Simplified calculation assuming magnetic field data
            beta_values = []
            for temp, density in zip(temperatures, densities):
                # Simplified beta calculation (actual would need magnetic field data)
                plasma_pressure = temp * density * 1.38e-23  # Boltzmann constant
                # Assuming magnetic pressure ~ 1e5 Pa (this would come from actual B-field data)
                magnetic_pressure = 1e5
                beta = plasma_pressure / magnetic_pressure
                beta_values.append(beta)
            
            if beta_values:
                metrics["avg_beta"] = np.mean(beta_values)
                metrics["max_beta"] = np.max(beta_values)
        
        # Window statistics
        metrics["window_size"] = len(self.data_window)
        metrics["data_rate"] = len(self.data_window) / (self.window_size if self.window_size > 0 else 1)
        
        return metrics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of recent values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        
        x_mean = np.mean(x)
        y_mean = np.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_batch_metrics(self, messages: List[StreamMessage]) -> Dict[str, Any]:
        """Calculate metrics for a batch of messages."""
        batch_metrics = {
            "batch_size": len(messages),
            "processing_time": datetime.now().isoformat(),
            "message_rate": len(messages),  # Messages per batch
        }
        
        # Calculate batch-level statistics
        temperatures = []
        densities = []
        
        for msg in messages:
            data = msg.data
            if "plasma_temperature" in data:
                temperatures.append(data["plasma_temperature"])
            if "plasma_density" in data:
                densities.append(data["plasma_density"])
        
        if temperatures:
            batch_metrics["batch_avg_temp"] = np.mean(temperatures)
            batch_metrics["batch_temp_range"] = np.max(temperatures) - np.min(temperatures)
        
        if densities:
            batch_metrics["batch_avg_density"] = np.mean(densities)
            batch_metrics["batch_density_range"] = np.max(densities) - np.min(densities)
        
        return batch_metrics
    
    def _check_alert_conditions(self, data: Dict[str, Any], metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        current_time = datetime.now()
        
        # Temperature alerts
        temp = data.get("plasma_temperature", 0)
        temp_threshold = self.alert_thresholds.get("temperature", {})
        
        if temp < temp_threshold.get("min", 0):
            alert_key = "low_temperature"
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    "type": "low_temperature",
                    "severity": "warning",
                    "message": f"Plasma temperature below threshold: {temp}K",
                    "value": temp,
                    "threshold": temp_threshold.get("min"),
                    "timestamp": current_time.isoformat()
                })
                self.last_alert_time[alert_key] = current_time
        
        if temp > temp_threshold.get("max", float('inf')):
            alert_key = "high_temperature"
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    "type": "high_temperature",
                    "severity": "critical",
                    "message": f"Plasma temperature above safe threshold: {temp}K",
                    "value": temp,
                    "threshold": temp_threshold.get("max"),
                    "timestamp": current_time.isoformat()
                })
                self.last_alert_time[alert_key] = current_time
        
        # Density alerts
        density = data.get("plasma_density", 0)
        density_threshold = self.alert_thresholds.get("density", {})
        
        if density < density_threshold.get("min", 0):
            alert_key = "low_density"
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    "type": "low_density",
                    "severity": "warning",
                    "message": f"Plasma density below threshold: {density}",
                    "value": density,
                    "threshold": density_threshold.get("min"),
                    "timestamp": current_time.isoformat()
                })
                self.last_alert_time[alert_key] = current_time
        
        # Triple product alerts
        if "avg_triple_product" in metrics:
            triple_product = metrics["avg_triple_product"]
            tp_threshold = self.alert_thresholds.get("triple_product", {})
            
            if triple_product > tp_threshold.get("ignition", 1e21):
                alert_key = "ignition_conditions"
                if self._should_send_alert(alert_key, current_time):
                    alerts.append({
                        "type": "ignition_conditions",
                        "severity": "info",
                        "message": f"Approaching ignition conditions: {triple_product:.2e}",
                        "value": triple_product,
                        "threshold": tp_threshold.get("ignition"),
                        "timestamp": current_time.isoformat()
                    })
                    self.last_alert_time[alert_key] = current_time
        
        # Trend alerts
        if "temp_trend" in metrics and abs(metrics["temp_trend"]) > self.alert_thresholds.get("temp_trend", 1000):
            alert_key = "temperature_trend"
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    "type": "temperature_trend",
                    "severity": "warning",
                    "message": f"Rapid temperature change detected: {metrics['temp_trend']:.2f}K/s",
                    "value": metrics["temp_trend"],
                    "timestamp": current_time.isoformat()
                })
                self.last_alert_time[alert_key] = current_time
        
        return alerts
    
    def _should_send_alert(self, alert_key: str, current_time: datetime) -> bool:
        """Check if alert should be sent (respecting cooldown)."""
        last_alert = self.last_alert_time.get(alert_key)
        if last_alert is None:
            return True
        
        return current_time - last_alert > self.alert_cooldown


class KafkaStreamProducer:
    """Kafka producer for streaming fusion data."""
    
    def __init__(self, 
                 bootstrap_servers: List[str],
                 topic: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kafka producer.
        
        Args:
            bootstrap_servers: Kafka broker addresses.
            topic: Kafka topic name.
            config: Additional producer configuration.
        """
        if not HAS_KAFKA:
            raise RuntimeError("Kafka library not available")
        
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.config = config or {}
        
        # Default producer configuration
        producer_config = {
            'bootstrap_servers': bootstrap_servers,
            'value_serializer': lambda x: json.dumps(x).encode('utf-8'),
            'key_serializer': lambda x: x.encode('utf-8') if isinstance(x, str) else None,
            'acks': 'all',  # Wait for all replicas
            'retries': 3,
            'batch_size': 16384,
            'linger_ms': 10,
            **self.config
        }
        
        self.producer = KafkaProducer(**producer_config)
        
        logger.info(f"KafkaStreamProducer initialized for topic: {topic}")
    
    async def send_message(self, message: StreamMessage, key: Optional[str] = None) -> bool:
        """
        Send message to Kafka topic.
        
        Args:
            message: Message to send.
            key: Optional message key for partitioning.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            future = self.producer.send(
                self.topic,
                value=message.to_dict(),
                key=key or message.message_id
            )
            
            # Wait for send completion
            record_metadata = future.get(timeout=10)
            
            logger.debug(f"Message sent: topic={record_metadata.topic}, "
                        f"partition={record_metadata.partition}, "
                        f"offset={record_metadata.offset}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to Kafka: {e}")
            return False
    
    async def send_batch(self, messages: List[StreamMessage]) -> int:
        """
        Send batch of messages to Kafka.
        
        Args:
            messages: List of messages to send.
            
        Returns:
            Number of successfully sent messages.
        """
        sent_count = 0
        
        for message in messages:
            if await self.send_message(message):
                sent_count += 1
        
        # Flush to ensure all messages are sent
        self.producer.flush()
        
        return sent_count
    
    def close(self):
        """Close Kafka producer."""
        if hasattr(self, 'producer'):
            self.producer.close()


class KafkaStreamConsumer:
    """Kafka consumer for streaming fusion data."""
    
    def __init__(self,
                 bootstrap_servers: List[str],
                 topic: str,
                 group_id: str,
                 processor: StreamProcessor,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kafka consumer.
        
        Args:
            bootstrap_servers: Kafka broker addresses.
            topic: Kafka topic name.
            group_id: Consumer group ID.
            processor: Stream processor for messages.
            config: Additional consumer configuration.
        """
        if not HAS_KAFKA:
            raise RuntimeError("Kafka library not available")
        
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.processor = processor
        self.config = config or {}
        
        # Default consumer configuration
        consumer_config = {
            'bootstrap_servers': bootstrap_servers,
            'group_id': group_id,
            'value_deserializer': lambda x: json.loads(x.decode('utf-8')),
            'key_deserializer': lambda x: x.decode('utf-8') if x else None,
            'auto_offset_reset': 'latest',
            'enable_auto_commit': True,
            'auto_commit_interval_ms': 1000,
            **self.config
        }
        
        self.consumer = KafkaConsumer(topic, **consumer_config)
        self.running = False
        
        logger.info(f"KafkaStreamConsumer initialized for topic: {topic}, group: {group_id}")
    
    async def start_consuming(self, batch_size: int = 100):
        """
        Start consuming messages from Kafka.
        
        Args:
            batch_size: Number of messages to process in each batch.
        """
        self.running = True
        batch = []
        
        logger.info("Starting Kafka message consumption")
        
        try:
            while self.running:
                # Poll for messages
                message_pack = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        try:
                            # Convert Kafka message to StreamMessage
                            stream_message = StreamMessage.from_dict(message.value)
                            batch.append(stream_message)
                            
                            # Process batch when full
                            if len(batch) >= batch_size:
                                await self._process_batch(batch)
                                batch = []
                                
                        except Exception as e:
                            logger.error(f"Error processing Kafka message: {e}")
                
                # Process remaining messages in batch
                if batch and not self.running:
                    await self._process_batch(batch)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in Kafka consumer: {e}")
        finally:
            self.consumer.close()
            logger.info("Kafka consumer stopped")
    
    async def _process_batch(self, batch: List[StreamMessage]):
        """Process a batch of messages."""
        try:
            processed_messages = await self.processor.process_batch(batch)
            logger.debug(f"Processed batch: {len(batch)} -> {len(processed_messages)} messages")
            
            # Here you could send processed messages to another topic or store them
            
        except Exception as e:
            logger.error(f"Error processing message batch: {e}")
    
    def stop(self):
        """Stop consuming messages."""
        self.running = False


class WebSocketStreamServer:
    """WebSocket server for real-time fusion data streaming."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize WebSocket server.
        
        Args:
            host: Server host.
            port: Server port.
        """
        if not HAS_WEBSOCKETS:
            raise RuntimeError("WebSockets library not available")
        
        self.host = host
        self.port = port
        self.clients = set()
        self.message_queue = asyncio.Queue()
        
        logger.info(f"WebSocketStreamServer initialized on {host}:{port}")
    
    async def register_client(self, websocket, path):
        """Register new WebSocket client."""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected: {websocket.remote_address}")
    
    async def broadcast_message(self, message: StreamMessage):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return
        
        message_data = json.dumps(message.to_dict())
        
        # Send to all clients
        disconnected_clients = set()
        for client in self.clients:
            try:
                await client.send(message_data)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
    
    async def start_server(self):
        """Start WebSocket server."""
        server = await websockets.serve(self.register_client, self.host, self.port)
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        
        # Start message processing loop
        asyncio.create_task(self._process_message_queue())
        
        await server.wait_closed()
    
    async def _process_message_queue(self):
        """Process messages from queue and broadcast to clients."""
        while True:
            try:
                message = await self.message_queue.get()
                await self.broadcast_message(message)
                self.message_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
                await asyncio.sleep(1)
    
    async def add_message(self, message: StreamMessage):
        """Add message to broadcast queue."""
        await self.message_queue.put(message)


class StreamingDataPipeline:
    """Complete streaming data pipeline for fusion analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize streaming pipeline.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.processor = FusionDataStreamProcessor(config.get("processor", {}))
        
        # Initialize components based on configuration
        self.kafka_producer = None
        self.kafka_consumer = None
        self.websocket_server = None
        
        if config.get("kafka", {}).get("enabled", False):
            kafka_config = config["kafka"]
            self.kafka_producer = KafkaStreamProducer(
                bootstrap_servers=kafka_config["bootstrap_servers"],
                topic=kafka_config["topic"],
                config=kafka_config.get("producer_config", {})
            )
            
            self.kafka_consumer = KafkaStreamConsumer(
                bootstrap_servers=kafka_config["bootstrap_servers"],
                topic=kafka_config["topic"],
                group_id=kafka_config.get("group_id", "fusion-analysis"),
                processor=self.processor,
                config=kafka_config.get("consumer_config", {})
            )
        
        if config.get("websocket", {}).get("enabled", False):
            ws_config = config["websocket"]
            self.websocket_server = WebSocketStreamServer(
                host=ws_config.get("host", "localhost"),
                port=ws_config.get("port", 8765)
            )
        
        logger.info("StreamingDataPipeline initialized")
    
    async def start(self):
        """Start the streaming pipeline."""
        tasks = []
        
        # Start Kafka consumer
        if self.kafka_consumer:
            tasks.append(asyncio.create_task(self.kafka_consumer.start_consuming()))
        
        # Start WebSocket server
        if self.websocket_server:
            tasks.append(asyncio.create_task(self.websocket_server.start_server()))
        
        if tasks:
            await asyncio.gather(*tasks)
        else:
            logger.warning("No streaming components enabled")
    
    async def send_data(self, data: Dict[str, Any], source: str = "fusion_sensor"):
        """
        Send data through the streaming pipeline.
        
        Args:
            data: Data to send.
            source: Data source identifier.
        """
        message = StreamMessage(data=data, source=source)
        
        # Process message
        processed_message = await self.processor.process_message(message)
        
        if processed_message:
            # Send to Kafka
            if self.kafka_producer:
                await self.kafka_producer.send_message(processed_message)
            
            # Send to WebSocket clients
            if self.websocket_server:
                await self.websocket_server.add_message(processed_message)
    
    def stop(self):
        """Stop the streaming pipeline."""
        if self.kafka_consumer:
            self.kafka_consumer.stop()
        
        if self.kafka_producer:
            self.kafka_producer.close()


def create_streaming_pipeline(config: Dict[str, Any]) -> StreamingDataPipeline:
    """
    Create streaming data pipeline.
    
    Args:
        config: Pipeline configuration.
        
    Returns:
        Configured streaming pipeline.
    """
    return StreamingDataPipeline(config)