"""
Event Streaming and Messaging System.

This module provides:
- Message queues for reliable event delivery
- Pub/Sub messaging patterns
- Event streaming and real-time processing
- Message persistence and replay
- Dead letter queues for failed messages
- Message routing and filtering
"""

import asyncio
import json
import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import queue
import logging
from concurrent.futures import ThreadPoolExecutor
import pickle

from . import Event, EventType

logger = logging.getLogger(__name__)


class MessageStatus(Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Message:
    """Message wrapper for events."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event: Event = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    routing_key: str = ""
    headers: Dict[str, Any] = field(default_factory=dict)
    ttl: Optional[int] = None  # Time to live in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        result = asdict(self)
        result['event'] = self.event.to_dict() if self.event else None
        result['timestamp'] = self.timestamp.isoformat()
        result['priority'] = self.priority.value
        result['status'] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        message_data = data.copy()
        if 'event' in message_data and message_data['event']:
            message_data['event'] = Event.from_dict(message_data['event'])
        message_data['timestamp'] = datetime.fromisoformat(message_data['timestamp'])
        message_data['priority'] = MessagePriority(message_data['priority'])
        message_data['status'] = MessageStatus(message_data['status'])
        return cls(**message_data)
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.ttl is None:
            return False
        
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl


class MessageQueue:
    """Priority message queue with persistence."""
    
    def __init__(self, name: str, max_size: int = 10000):
        """
        Initialize message queue.
        
        Args:
            name: Queue name.
            max_size: Maximum queue size.
        """
        self.name = name
        self.max_size = max_size
        
        # Priority queues (higher number = higher priority)
        self.queues: Dict[MessagePriority, queue.PriorityQueue] = {
            priority: queue.PriorityQueue() for priority in MessagePriority
        }
        
        self.dead_letter_queue: List[Message] = []
        self.processing_messages: Dict[str, Message] = {}
        
        self.lock = threading.Lock()
        self.stats = {
            "messages_sent": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "messages_dead_letter": 0
        }
        
        logger.info(f"MessageQueue initialized: {name}")
    
    def enqueue(self, message: Message) -> bool:
        """
        Enqueue message.
        
        Args:
            message: Message to enqueue.
            
        Returns:
            True if successful.
        """
        try:
            with self.lock:
                # Check if message is expired
                if message.is_expired():
                    logger.warning(f"Message expired, not queuing: {message.id}")
                    return False
                
                # Check queue size
                total_size = sum(q.qsize() for q in self.queues.values())
                if total_size >= self.max_size:
                    logger.warning(f"Queue {self.name} is full, dropping message: {message.id}")
                    return False
                
                # Enqueue with priority (negative for correct priority order)
                priority_queue = self.queues[message.priority]
                priority_queue.put((-message.priority.value, time.time(), message))
                
                self.stats["messages_sent"] += 1
                
                logger.debug(f"Message enqueued: {message.id} with priority {message.priority.name}")
                return True
                
        except Exception as e:
            logger.error(f"Error enqueueing message: {e}")
            return False
    
    def dequeue(self, timeout: float = 1.0) -> Optional[Message]:
        """
        Dequeue message with highest priority.
        
        Args:
            timeout: Timeout for dequeue operation.
            
        Returns:
            Message or None.
        """
        try:
            # Try each priority queue from highest to lowest
            for priority in sorted(MessagePriority, key=lambda p: p.value, reverse=True):
                priority_queue = self.queues[priority]
                
                if not priority_queue.empty():
                    try:
                        _, _, message = priority_queue.get_nowait()
                        
                        # Check if message is expired
                        if message.is_expired():
                            logger.warning(f"Dequeued expired message: {message.id}")
                            continue
                        
                        # Mark as processing
                        message.status = MessageStatus.PROCESSING
                        with self.lock:
                            self.processing_messages[message.id] = message
                        
                        logger.debug(f"Message dequeued: {message.id}")
                        return message
                        
                    except queue.Empty:
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error dequeuing message: {e}")
            return None
    
    def ack(self, message_id: str) -> bool:
        """
        Acknowledge message processing.
        
        Args:
            message_id: Message ID to acknowledge.
            
        Returns:
            True if successful.
        """
        try:
            with self.lock:
                if message_id in self.processing_messages:
                    message = self.processing_messages.pop(message_id)
                    message.status = MessageStatus.COMPLETED
                    self.stats["messages_processed"] += 1
                    
                    logger.debug(f"Message acknowledged: {message_id}")
                    return True
            
            logger.warning(f"Message not found for ack: {message_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging message: {e}")
            return False
    
    def nack(self, message_id: str, requeue: bool = True) -> bool:
        """
        Negative acknowledge message processing.
        
        Args:
            message_id: Message ID to nack.
            requeue: Whether to requeue the message.
            
        Returns:
            True if successful.
        """
        try:
            with self.lock:
                if message_id in self.processing_messages:
                    message = self.processing_messages.pop(message_id)
                    message.retry_count += 1
                    
                    if requeue and message.retry_count <= message.max_retries:
                        # Requeue with delay
                        message.status = MessageStatus.PENDING
                        self.enqueue(message)
                        logger.debug(f"Message requeued: {message_id} (retry {message.retry_count})")
                    else:
                        # Move to dead letter queue
                        message.status = MessageStatus.DEAD_LETTER
                        self.dead_letter_queue.append(message)
                        self.stats["messages_dead_letter"] += 1
                        logger.warning(f"Message moved to dead letter queue: {message_id}")
                    
                    self.stats["messages_failed"] += 1
                    return True
            
            logger.warning(f"Message not found for nack: {message_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error nacking message: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            queue_sizes = {
                priority.name: self.queues[priority].qsize() 
                for priority in MessagePriority
            }
            
            return {
                "name": self.name,
                "queue_sizes": queue_sizes,
                "total_queued": sum(queue_sizes.values()),
                "processing": len(self.processing_messages),
                "dead_letter": len(self.dead_letter_queue),
                "stats": self.stats.copy()
            }
    
    def get_dead_letter_messages(self) -> List[Message]:
        """Get dead letter messages."""
        with self.lock:
            return self.dead_letter_queue.copy()
    
    def requeue_dead_letter(self, message_id: str) -> bool:
        """
        Requeue message from dead letter queue.
        
        Args:
            message_id: Message ID to requeue.
            
        Returns:
            True if successful.
        """
        try:
            with self.lock:
                for i, message in enumerate(self.dead_letter_queue):
                    if message.id == message_id:
                        # Reset message
                        message.status = MessageStatus.PENDING
                        message.retry_count = 0
                        
                        # Remove from dead letter and requeue
                        self.dead_letter_queue.pop(i)
                        return self.enqueue(message)
            
            return False
            
        except Exception as e:
            logger.error(f"Error requeuing dead letter message: {e}")
            return False


class MessageBroker:
    """Message broker for routing and delivery."""
    
    def __init__(self):
        """Initialize message broker."""
        self.queues: Dict[str, MessageQueue] = {}
        self.topics: Dict[str, Set[str]] = {}  # topic -> queue names
        self.routing_rules: List[Dict[str, Any]] = []
        
        self.running = False
        self.worker_threads: List[threading.Thread] = []
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        logger.info("MessageBroker initialized")
    
    def create_queue(self, name: str, max_size: int = 10000) -> MessageQueue:
        """
        Create message queue.
        
        Args:
            name: Queue name.
            max_size: Maximum queue size.
            
        Returns:
            Created queue.
        """
        if name in self.queues:
            return self.queues[name]
        
        queue_obj = MessageQueue(name, max_size)
        self.queues[name] = queue_obj
        
        logger.info(f"Queue created: {name}")
        return queue_obj
    
    def delete_queue(self, name: str) -> bool:
        """
        Delete message queue.
        
        Args:
            name: Queue name.
            
        Returns:
            True if successful.
        """
        if name in self.queues:
            del self.queues[name]
            
            # Remove from topics
            for topic_queues in self.topics.values():
                topic_queues.discard(name)
            
            logger.info(f"Queue deleted: {name}")
            return True
        
        return False
    
    def bind_queue(self, queue_name: str, topic: str):
        """
        Bind queue to topic.
        
        Args:
            queue_name: Queue name.
            topic: Topic name.
        """
        if topic not in self.topics:
            self.topics[topic] = set()
        
        self.topics[topic].add(queue_name)
        logger.info(f"Queue {queue_name} bound to topic {topic}")
    
    def unbind_queue(self, queue_name: str, topic: str):
        """
        Unbind queue from topic.
        
        Args:
            queue_name: Queue name.
            topic: Topic name.
        """
        if topic in self.topics:
            self.topics[topic].discard(queue_name)
            
            # Remove empty topics
            if not self.topics[topic]:
                del self.topics[topic]
        
        logger.info(f"Queue {queue_name} unbound from topic {topic}")
    
    def add_routing_rule(self, pattern: str, queue_name: str):
        """
        Add message routing rule.
        
        Args:
            pattern: Routing pattern (supports wildcards).
            queue_name: Target queue name.
        """
        rule = {
            "pattern": pattern,
            "queue": queue_name
        }
        self.routing_rules.append(rule)
        logger.info(f"Routing rule added: {pattern} -> {queue_name}")
    
    def publish(self, event: Event, routing_key: str = "", priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """
        Publish event to broker.
        
        Args:
            event: Event to publish.
            routing_key: Routing key for message.
            priority: Message priority.
            
        Returns:
            True if successful.
        """
        try:
            # Create message
            message = Message(
                event=event,
                routing_key=routing_key,
                priority=priority
            )
            
            # Route to appropriate queues
            target_queues = self._route_message(message)
            
            if not target_queues:
                logger.warning(f"No queues found for routing key: {routing_key}")
                return False
            
            # Enqueue to all target queues
            success_count = 0
            for queue_name in target_queues:
                if queue_name in self.queues:
                    if self.queues[queue_name].enqueue(message):
                        success_count += 1
            
            logger.debug(f"Message published to {success_count}/{len(target_queues)} queues")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
    
    def subscribe(self, queue_name: str, callback: Callable[[Message], bool], 
                 auto_ack: bool = True) -> bool:
        """
        Subscribe to queue.
        
        Args:
            queue_name: Queue name to subscribe to.
            callback: Callback function for messages.
            auto_ack: Whether to auto-acknowledge messages.
            
        Returns:
            True if successful.
        """
        if queue_name not in self.queues:
            logger.error(f"Queue not found: {queue_name}")
            return False
        
        # Start worker thread for this subscription
        worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(queue_name, callback, auto_ack)
        )
        worker_thread.daemon = True
        worker_thread.start()
        
        self.worker_threads.append(worker_thread)
        logger.info(f"Subscribed to queue: {queue_name}")
        return True
    
    def start(self):
        """Start message broker."""
        self.running = True
        logger.info("MessageBroker started")
    
    def stop(self):
        """Stop message broker."""
        self.running = False
        
        # Wait for worker threads
        for thread in self.worker_threads:
            thread.join(timeout=5)
        
        self.executor.shutdown(wait=True)
        logger.info("MessageBroker stopped")
    
    def get_broker_stats(self) -> Dict[str, Any]:
        """Get broker statistics."""
        queue_stats = {
            name: queue_obj.get_stats() 
            for name, queue_obj in self.queues.items()
        }
        
        topic_stats = {
            topic: list(queue_names) 
            for topic, queue_names in self.topics.items()
        }
        
        return {
            "queues": queue_stats,
            "topics": topic_stats,
            "routing_rules": len(self.routing_rules),
            "worker_threads": len(self.worker_threads)
        }
    
    def _route_message(self, message: Message) -> Set[str]:
        """
        Route message to appropriate queues.
        
        Args:
            message: Message to route.
            
        Returns:
            Set of queue names.
        """
        target_queues = set()
        routing_key = message.routing_key
        
        # Check topic bindings
        for topic, queue_names in self.topics.items():
            if self._match_routing_key(routing_key, topic):
                target_queues.update(queue_names)
        
        # Check routing rules
        for rule in self.routing_rules:
            if self._match_routing_key(routing_key, rule["pattern"]):
                target_queues.add(rule["queue"])
        
        # Default routing by event type
        if not target_queues and message.event:
            event_type_queue = f"events.{message.event.event_type.value}"
            if event_type_queue in self.queues:
                target_queues.add(event_type_queue)
        
        return target_queues
    
    def _match_routing_key(self, routing_key: str, pattern: str) -> bool:
        """
        Check if routing key matches pattern.
        
        Args:
            routing_key: Routing key to check.
            pattern: Pattern to match against.
            
        Returns:
            True if matches.
        """
        # Simple wildcard matching
        if pattern == "*":
            return True
        
        if pattern == routing_key:
            return True
        
        # Partial wildcard matching (simplified)
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return routing_key.startswith(prefix)
        
        if pattern.startswith("*"):
            suffix = pattern[1:]
            return routing_key.endswith(suffix)
        
        return False
    
    def _worker_loop(self, queue_name: str, callback: Callable[[Message], bool], auto_ack: bool):
        """
        Worker loop for processing messages.
        
        Args:
            queue_name: Queue name to process.
            callback: Callback function.
            auto_ack: Whether to auto-acknowledge.
        """
        queue_obj = self.queues[queue_name]
        
        while self.running:
            try:
                message = queue_obj.dequeue(timeout=1.0)
                
                if message is None:
                    continue
                
                # Process message
                try:
                    success = callback(message)
                    
                    if auto_ack:
                        if success:
                            queue_obj.ack(message.id)
                        else:
                            queue_obj.nack(message.id)
                
                except Exception as e:
                    logger.error(f"Error in message callback: {e}")
                    if auto_ack:
                        queue_obj.nack(message.id)
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1)


class EventStream:
    """Event streaming for real-time processing."""
    
    def __init__(self, name: str, buffer_size: int = 1000):
        """
        Initialize event stream.
        
        Args:
            name: Stream name.
            buffer_size: Stream buffer size.
        """
        self.name = name
        self.buffer_size = buffer_size
        
        self.events: List[Event] = []
        self.subscribers: List[Callable[[Event], None]] = []
        self.filters: List[Callable[[Event], bool]] = []
        
        self.lock = threading.Lock()
        self.stats = {
            "events_published": 0,
            "events_consumed": 0
        }
        
        logger.info(f"EventStream initialized: {name}")
    
    def publish(self, event: Event):
        """
        Publish event to stream.
        
        Args:
            event: Event to publish.
        """
        with self.lock:
            # Add to buffer
            self.events.append(event)
            
            # Maintain buffer size
            if len(self.events) > self.buffer_size:
                self.events = self.events[-self.buffer_size:]
            
            self.stats["events_published"] += 1
        
        # Notify subscribers
        self._notify_subscribers(event)
    
    def subscribe(self, callback: Callable[[Event], None], 
                 event_filter: Callable[[Event], bool] = None):
        """
        Subscribe to event stream.
        
        Args:
            callback: Callback function for events.
            event_filter: Optional filter function.
        """
        self.subscribers.append(callback)
        
        if event_filter:
            self.filters.append(event_filter)
        
        logger.info(f"Subscriber added to stream: {self.name}")
    
    def get_events(self, from_index: int = 0, count: int = None) -> List[Event]:
        """
        Get events from stream.
        
        Args:
            from_index: Starting index.
            count: Number of events to return.
            
        Returns:
            List of events.
        """
        with self.lock:
            events = self.events[from_index:]
            
            if count is not None:
                events = events[:count]
            
            self.stats["events_consumed"] += len(events)
            return events
    
    def get_latest_events(self, count: int = 10) -> List[Event]:
        """
        Get latest events from stream.
        
        Args:
            count: Number of events to return.
            
        Returns:
            List of latest events.
        """
        with self.lock:
            return self.events[-count:] if len(self.events) > count else self.events.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        with self.lock:
            return {
                "name": self.name,
                "buffer_size": self.buffer_size,
                "current_events": len(self.events),
                "subscribers": len(self.subscribers),
                "filters": len(self.filters),
                "stats": self.stats.copy()
            }
    
    def _notify_subscribers(self, event: Event):
        """
        Notify subscribers of new event.
        
        Args:
            event: Event to notify about.
        """
        for i, callback in enumerate(self.subscribers):
            try:
                # Apply filter if available
                if i < len(self.filters):
                    if not self.filters[i](event):
                        continue
                
                callback(event)
                
            except Exception as e:
                logger.error(f"Error in stream subscriber: {e}")


# Global messaging components
_message_broker = None
_event_streams = {}


def get_message_broker() -> MessageBroker:
    """Get global message broker."""
    global _message_broker
    if _message_broker is None:
        _message_broker = MessageBroker()
    return _message_broker


def get_event_stream(name: str, buffer_size: int = 1000) -> EventStream:
    """
    Get or create event stream.
    
    Args:
        name: Stream name.
        buffer_size: Buffer size.
        
    Returns:
        Event stream.
    """
    global _event_streams
    
    if name not in _event_streams:
        _event_streams[name] = EventStream(name, buffer_size)
    
    return _event_streams[name]


def start_messaging():
    """Start messaging system."""
    broker = get_message_broker()
    broker.start()
    logger.info("Messaging system started")


def stop_messaging():
    """Stop messaging system."""
    broker = get_message_broker()
    broker.stop()
    logger.info("Messaging system stopped")