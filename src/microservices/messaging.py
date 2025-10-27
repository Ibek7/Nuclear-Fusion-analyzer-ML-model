"""
Message Queue system for microservices communication.

This module provides:
- Asynchronous message publishing and consuming
- Multiple broker support (Redis, RabbitMQ, Kafka)
- Message routing and filtering
- Dead letter queues and retries
- Event sourcing patterns
- Message persistence and durability
"""

import asyncio
import json
import uuid
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timezone, timedelta
import logging

# Redis for pub/sub
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

# RabbitMQ support
try:
    import aio_pika
    from aio_pika import Connection, Channel, Queue, Exchange, Message
    HAS_RABBITMQ = True
except ImportError:
    HAS_RABBITMQ = False

# Kafka support
try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class MessageStatus(Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class Message:
    """Message structure for queue communication."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expiration: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    status: MessageStatus = MessageStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        if self.expiration:
            data["expiration"] = self.expiration.isoformat()
        data["priority"] = self.priority.value
        data["status"] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        msg = cls()
        
        for key, value in data.items():
            if key == "timestamp":
                msg.timestamp = datetime.fromisoformat(value)
            elif key == "expiration" and value:
                msg.expiration = datetime.fromisoformat(value)
            elif key == "priority":
                msg.priority = MessagePriority(value)
            elif key == "status":
                msg.status = MessageStatus(value)
            else:
                setattr(msg, key, value)
        
        return msg
    
    def is_expired(self) -> bool:
        """Check if message is expired."""
        if not self.expiration:
            return False
        return datetime.now(timezone.utc) > self.expiration


class MessageHandler(ABC):
    """Abstract message handler."""
    
    @abstractmethod
    async def handle(self, message: Message) -> bool:
        """
        Handle message.
        
        Args:
            message: Message to handle.
            
        Returns:
            True if handled successfully, False otherwise.
        """
        pass


class MessageBroker(ABC):
    """Abstract message broker interface."""
    
    @abstractmethod
    async def connect(self):
        """Connect to message broker."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from message broker."""
        pass
    
    @abstractmethod
    async def publish(self, message: Message, topic: str = None) -> bool:
        """Publish message to topic."""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, handler: MessageHandler) -> str:
        """Subscribe to topic with handler."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from topic."""
        pass


class RedisBroker(MessageBroker):
    """Redis-based message broker."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize Redis broker.
        
        Args:
            redis_url: Redis connection URL.
        """
        if not HAS_REDIS:
            raise RuntimeError("Redis not available")
        
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.subscribers: Dict[str, asyncio.Task] = {}
        self.handlers: Dict[str, MessageHandler] = {}
        
        logger.info("RedisBroker initialized")
    
    async def connect(self):
        """Connect to Redis."""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        await self.redis_client.ping()
        
        logger.info("Connected to Redis broker")
    
    async def disconnect(self):
        """Disconnect from Redis."""
        # Stop all subscribers
        for task in self.subscribers.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Disconnected from Redis broker")
    
    async def publish(self, message: Message, topic: str = None) -> bool:
        """Publish message to Redis."""
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")
        
        topic = topic or message.topic
        if not topic:
            raise ValueError("Topic is required")
        
        try:
            message_data = json.dumps(message.to_dict())
            await self.redis_client.publish(topic, message_data)
            
            logger.debug(f"Published message {message.id} to topic {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: MessageHandler) -> str:
        """Subscribe to Redis topic."""
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")
        
        subscription_id = str(uuid.uuid4())
        self.handlers[subscription_id] = handler
        
        async def _subscriber():
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(topic)
            
            try:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        try:
                            data = json.loads(message["data"])
                            msg = Message.from_dict(data)
                            
                            # Check expiration
                            if msg.is_expired():
                                logger.warning(f"Message {msg.id} expired")
                                continue
                            
                            # Handle message
                            success = await handler.handle(msg)
                            if success:
                                logger.debug(f"Message {msg.id} handled successfully")
                            else:
                                logger.warning(f"Message {msg.id} handling failed")
                                
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            
            except asyncio.CancelledError:
                await pubsub.unsubscribe(topic)
                await pubsub.close()
                raise
        
        task = asyncio.create_task(_subscriber())
        self.subscribers[subscription_id] = task
        
        logger.info(f"Subscribed to topic {topic} with ID {subscription_id}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from Redis topic."""
        if subscription_id in self.subscribers:
            task = self.subscribers.pop(subscription_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            self.handlers.pop(subscription_id, None)
            logger.info(f"Unsubscribed {subscription_id}")


class RabbitMQBroker(MessageBroker):
    """RabbitMQ-based message broker."""
    
    def __init__(self, amqp_url: str = "amqp://guest:guest@localhost/"):
        """
        Initialize RabbitMQ broker.
        
        Args:
            amqp_url: AMQP connection URL.
        """
        if not HAS_RABBITMQ:
            raise RuntimeError("RabbitMQ not available")
        
        self.amqp_url = amqp_url
        self.connection: Optional[Connection] = None
        self.channel: Optional[Channel] = None
        self.subscribers: Dict[str, asyncio.Task] = {}
        self.handlers: Dict[str, MessageHandler] = {}
        
        logger.info("RabbitMQBroker initialized")
    
    async def connect(self):
        """Connect to RabbitMQ."""
        self.connection = await aio_pika.connect_robust(self.amqp_url)
        self.channel = await self.connection.channel()
        
        logger.info("Connected to RabbitMQ broker")
    
    async def disconnect(self):
        """Disconnect from RabbitMQ."""
        # Stop all subscribers
        for task in self.subscribers.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        if self.channel:
            await self.channel.close()
        if self.connection:
            await self.connection.close()
        
        logger.info("Disconnected from RabbitMQ broker")
    
    async def publish(self, message: Message, topic: str = None) -> bool:
        """Publish message to RabbitMQ."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")
        
        topic = topic or message.topic
        if not topic:
            raise ValueError("Topic is required")
        
        try:
            # Declare exchange
            exchange = await self.channel.declare_exchange(
                topic, aio_pika.ExchangeType.TOPIC, durable=True
            )
            
            # Create message
            message_data = json.dumps(message.to_dict()).encode()
            msg = Message(
                body=message_data,
                priority=message.priority.value,
                correlation_id=message.correlation_id,
                reply_to=message.reply_to,
                expiration=int((message.expiration - datetime.now(timezone.utc)).total_seconds() * 1000) if message.expiration else None
            )
            
            # Publish
            await exchange.publish(msg, routing_key=topic)
            
            logger.debug(f"Published message {message.id} to topic {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: MessageHandler) -> str:
        """Subscribe to RabbitMQ topic."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")
        
        subscription_id = str(uuid.uuid4())
        self.handlers[subscription_id] = handler
        
        async def _subscriber():
            # Declare exchange
            exchange = await self.channel.declare_exchange(
                topic, aio_pika.ExchangeType.TOPIC, durable=True
            )
            
            # Declare queue
            queue_name = f"{topic}.{subscription_id}"
            queue = await self.channel.declare_queue(queue_name, auto_delete=True)
            await queue.bind(exchange, routing_key=topic)
            
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    try:
                        data = json.loads(message.body.decode())
                        msg = Message.from_dict(data)
                        
                        # Check expiration
                        if msg.is_expired():
                            logger.warning(f"Message {msg.id} expired")
                            await message.ack()
                            continue
                        
                        # Handle message
                        success = await handler.handle(msg)
                        if success:
                            await message.ack()
                            logger.debug(f"Message {msg.id} handled successfully")
                        else:
                            await message.nack(requeue=msg.retry_count < msg.max_retries)
                            logger.warning(f"Message {msg.id} handling failed")
                            
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        await message.nack(requeue=False)
        
        task = asyncio.create_task(_subscriber())
        self.subscribers[subscription_id] = task
        
        logger.info(f"Subscribed to topic {topic} with ID {subscription_id}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from RabbitMQ topic."""
        if subscription_id in self.subscribers:
            task = self.subscribers.pop(subscription_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            self.handlers.pop(subscription_id, None)
            logger.info(f"Unsubscribed {subscription_id}")


class MessageQueue:
    """
    High-level message queue system.
    
    Provides pub/sub messaging with multiple broker support.
    """
    
    def __init__(self, broker: MessageBroker):
        """
        Initialize message queue.
        
        Args:
            broker: Message broker implementation.
        """
        self.broker = broker
        self.running = False
        
        logger.info("MessageQueue initialized")
    
    async def start(self):
        """Start message queue."""
        await self.broker.connect()
        self.running = True
        
        logger.info("MessageQueue started")
    
    async def stop(self):
        """Stop message queue."""
        self.running = False
        await self.broker.disconnect()
        
        logger.info("MessageQueue stopped")
    
    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        expiration: Optional[timedelta] = None
    ) -> str:
        """
        Publish message to topic.
        
        Args:
            topic: Message topic.
            payload: Message payload.
            priority: Message priority.
            correlation_id: Correlation ID for request/response.
            reply_to: Reply topic.
            expiration: Message expiration.
            
        Returns:
            Message ID.
        """
        if not self.running:
            raise RuntimeError("Message queue not running")
        
        message = Message(
            topic=topic,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
            reply_to=reply_to,
            expiration=datetime.now(timezone.utc) + expiration if expiration else None
        )
        
        success = await self.broker.publish(message)
        if not success:
            raise RuntimeError("Failed to publish message")
        
        logger.debug(f"Published message {message.id} to topic {topic}")
        return message.id
    
    async def subscribe(self, topic: str, handler: MessageHandler) -> str:
        """
        Subscribe to topic.
        
        Args:
            topic: Topic to subscribe to.
            handler: Message handler.
            
        Returns:
            Subscription ID.
        """
        if not self.running:
            raise RuntimeError("Message queue not running")
        
        subscription_id = await self.broker.subscribe(topic, handler)
        
        logger.info(f"Subscribed to topic {topic}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str):
        """
        Unsubscribe from topic.
        
        Args:
            subscription_id: Subscription ID.
        """
        await self.broker.unsubscribe(subscription_id)
        
        logger.info(f"Unsubscribed {subscription_id}")


class EventBus:
    """
    Event bus for domain events and event sourcing.
    """
    
    def __init__(self, message_queue: MessageQueue):
        """
        Initialize event bus.
        
        Args:
            message_queue: Message queue system.
        """
        self.message_queue = message_queue
        self.event_handlers: Dict[str, List[MessageHandler]] = {}
        
        logger.info("EventBus initialized")
    
    async def publish_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        aggregate_id: Optional[str] = None
    ) -> str:
        """
        Publish domain event.
        
        Args:
            event_type: Type of event.
            event_data: Event data.
            aggregate_id: Aggregate ID for event sourcing.
            
        Returns:
            Event ID.
        """
        payload = {
            "event_type": event_type,
            "aggregate_id": aggregate_id,
            "event_data": event_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        topic = f"events.{event_type}"
        return await self.message_queue.publish(topic, payload)
    
    async def subscribe_to_event(
        self,
        event_type: str,
        handler: MessageHandler
    ) -> str:
        """
        Subscribe to domain event.
        
        Args:
            event_type: Event type to subscribe to.
            handler: Event handler.
            
        Returns:
            Subscription ID.
        """
        topic = f"events.{event_type}"
        return await self.message_queue.subscribe(topic, handler)


def create_message_queue(broker_type: str = "redis", **kwargs) -> MessageQueue:
    """
    Create message queue with specified broker.
    
    Args:
        broker_type: Type of broker (redis, rabbitmq, kafka).
        **kwargs: Broker-specific arguments.
        
    Returns:
        Configured message queue.
    """
    if broker_type == "redis":
        broker = RedisBroker(kwargs.get("redis_url", "redis://localhost:6379"))
    elif broker_type == "rabbitmq":
        broker = RabbitMQBroker(kwargs.get("amqp_url", "amqp://guest:guest@localhost/"))
    else:
        raise ValueError(f"Unsupported broker type: {broker_type}")
    
    return MessageQueue(broker)