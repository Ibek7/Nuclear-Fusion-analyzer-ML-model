"""
Event-Driven Architecture System for Nuclear Fusion Analyzer.

This module provides:
- Event sourcing with event store
- Message queuing and pub/sub patterns
- Event handlers and subscribers
- CQRS (Command Query Responsibility Segregation) implementation
- Distributed event processing
- Event replay and time-travel debugging
- Saga pattern for distributed transactions
- Event versioning and schema evolution
"""

import asyncio
import json
import uuid
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Type, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
import pickle
import copy

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for the fusion analyzer system."""
    # System events
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    SYSTEM_ERROR = "system.error"
    
    # Fusion analysis events
    ANALYSIS_STARTED = "fusion.analysis.started"
    ANALYSIS_COMPLETED = "fusion.analysis.completed"
    ANALYSIS_FAILED = "fusion.analysis.failed"
    
    # Plasma events
    PLASMA_TEMPERATURE_CHANGED = "fusion.plasma.temperature_changed"
    PLASMA_DENSITY_CHANGED = "fusion.plasma.density_changed"
    PLASMA_INSTABILITY_DETECTED = "fusion.plasma.instability_detected"
    
    # Safety events
    SAFETY_THRESHOLD_EXCEEDED = "fusion.safety.threshold_exceeded"
    EMERGENCY_SHUTDOWN = "fusion.safety.emergency_shutdown"
    SAFETY_SYSTEM_ARMED = "fusion.safety.system_armed"
    
    # ML model events
    MODEL_TRAINED = "ml.model.trained"
    MODEL_DEPLOYED = "ml.model.deployed"
    PREDICTION_MADE = "ml.prediction.made"
    MODEL_PERFORMANCE_DEGRADED = "ml.model.performance_degraded"
    
    # User events
    USER_LOGGED_IN = "user.logged_in"
    USER_LOGGED_OUT = "user.logged_out"
    USER_ACTION = "user.action"
    
    # Configuration events
    CONFIG_CHANGED = "config.changed"
    FEATURE_FLAG_TOGGLED = "config.feature_flag.toggled"
    
    # Monitoring events
    ALERT_FIRED = "monitoring.alert.fired"
    ALERT_RESOLVED = "monitoring.alert.resolved"
    SLA_VIOLATED = "monitoring.sla.violated"
    
    # Data events
    DATA_IMPORTED = "data.imported"
    DATA_EXPORTED = "data.exported"
    DATA_CORRUPTED = "data.corrupted"


@dataclass
class Event:
    """Base event class."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.SYSTEM_STARTED
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
    source: str = "fusion_analyzer"
    aggregate_id: Optional[str] = None
    aggregate_type: Optional[str] = None
    sequence_number: Optional[int] = None
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        event_data = copy.deepcopy(data)
        event_data['event_type'] = EventType(event_data['event_type'])
        event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
        return cls(**event_data)


@dataclass
class Command:
    """Base command class for CQRS."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    command_type: str = "base_command"
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    """Base query class for CQRS."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_type: str = "base_query"
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if handler can process the event."""
        pass
    
    @abstractmethod
    async def handle(self, event: Event) -> Optional[List[Event]]:
        """Handle the event and optionally return new events."""
        pass
    
    def get_handler_name(self) -> str:
        """Get handler name."""
        return self.__class__.__name__


class CommandHandler(ABC):
    """Abstract base class for command handlers."""
    
    @abstractmethod
    def can_handle(self, command: Command) -> bool:
        """Check if handler can process the command."""
        pass
    
    @abstractmethod
    async def handle(self, command: Command) -> List[Event]:
        """Handle the command and return resulting events."""
        pass


class QueryHandler(ABC):
    """Abstract base class for query handlers."""
    
    @abstractmethod
    def can_handle(self, query: Query) -> bool:
        """Check if handler can process the query."""
        pass
    
    @abstractmethod
    async def handle(self, query: Query) -> Any:
        """Handle the query and return result."""
        pass


class EventStore:
    """Event store for persisting events."""
    
    def __init__(self, storage_path: str = "events.json"):
        """
        Initialize event store.
        
        Args:
            storage_path: Path to storage file.
        """
        self.storage_path = storage_path
        self.events: List[Event] = []
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.sequence_counter = 0
        self.lock = threading.Lock()
        
        self._load_events()
        logger.info("EventStore initialized")
    
    def append_event(self, event: Event) -> bool:
        """
        Append event to store.
        
        Args:
            event: Event to append.
            
        Returns:
            True if successful.
        """
        try:
            with self.lock:
                self.sequence_counter += 1
                event.sequence_number = self.sequence_counter
                self.events.append(event)
                
                self._save_events()
                
                logger.debug(f"Event appended: {event.event_type.value}")
                return True
                
        except Exception as e:
            logger.error(f"Error appending event: {e}")
            return False
    
    def get_events(self, aggregate_id: str = None, 
                  event_type: EventType = None,
                  from_sequence: int = None,
                  to_sequence: int = None) -> List[Event]:
        """
        Get events from store.
        
        Args:
            aggregate_id: Filter by aggregate ID.
            event_type: Filter by event type.
            from_sequence: Starting sequence number.
            to_sequence: Ending sequence number.
            
        Returns:
            List of events.
        """
        with self.lock:
            filtered_events = self.events.copy()
        
        if aggregate_id:
            filtered_events = [e for e in filtered_events if e.aggregate_id == aggregate_id]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if from_sequence is not None:
            filtered_events = [e for e in filtered_events 
                             if e.sequence_number and e.sequence_number >= from_sequence]
        
        if to_sequence is not None:
            filtered_events = [e for e in filtered_events 
                             if e.sequence_number and e.sequence_number <= to_sequence]
        
        return filtered_events
    
    def get_latest_events(self, count: int = 100) -> List[Event]:
        """
        Get latest events.
        
        Args:
            count: Number of events to return.
            
        Returns:
            List of latest events.
        """
        with self.lock:
            return self.events[-count:] if len(self.events) > count else self.events.copy()
    
    def save_snapshot(self, aggregate_id: str, aggregate_type: str, 
                     state: Dict[str, Any], sequence_number: int):
        """
        Save aggregate snapshot.
        
        Args:
            aggregate_id: Aggregate ID.
            aggregate_type: Aggregate type.
            state: Aggregate state.
            sequence_number: Sequence number of last applied event.
        """
        snapshot_key = f"{aggregate_type}:{aggregate_id}"
        self.snapshots[snapshot_key] = {
            "state": state,
            "sequence_number": sequence_number,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_snapshot(self, aggregate_id: str, aggregate_type: str) -> Optional[Dict[str, Any]]:
        """
        Get aggregate snapshot.
        
        Args:
            aggregate_id: Aggregate ID.
            aggregate_type: Aggregate type.
            
        Returns:
            Snapshot data or None.
        """
        snapshot_key = f"{aggregate_type}:{aggregate_id}"
        return self.snapshots.get(snapshot_key)
    
    def replay_events(self, from_sequence: int = 0) -> List[Event]:
        """
        Replay events from sequence number.
        
        Args:
            from_sequence: Starting sequence number.
            
        Returns:
            List of events to replay.
        """
        return self.get_events(from_sequence=from_sequence)
    
    def get_event_count(self) -> int:
        """Get total event count."""
        with self.lock:
            return len(self.events)
    
    def _load_events(self):
        """Load events from storage."""
        try:
            import os
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.events = [Event.from_dict(event_data) for event_data in data.get('events', [])]
                    self.snapshots = data.get('snapshots', {})
                    self.sequence_counter = data.get('sequence_counter', 0)
                    
                logger.info(f"Loaded {len(self.events)} events from storage")
        except Exception as e:
            logger.warning(f"Could not load events from storage: {e}")
    
    def _save_events(self):
        """Save events to storage."""
        try:
            data = {
                'events': [event.to_dict() for event in self.events],
                'snapshots': self.snapshots,
                'sequence_counter': self.sequence_counter
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving events to storage: {e}")


class EventBus:
    """Event bus for publishing and subscribing to events."""
    
    def __init__(self, event_store: EventStore):
        """
        Initialize event bus.
        
        Args:
            event_store: Event store instance.
        """
        self.event_store = event_store
        self.handlers: List[EventHandler] = []
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.middleware: List[Callable] = []
        
        self.processing_queue = queue.Queue()
        self.is_processing = False
        self.processor_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("EventBus initialized")
    
    def start(self):
        """Start event processing."""
        if not self.is_processing:
            self.is_processing = True
            self.processor_thread = threading.Thread(target=self._process_events)
            self.processor_thread.daemon = True
            self.processor_thread.start()
            logger.info("EventBus started")
    
    def stop(self):
        """Stop event processing."""
        self.is_processing = False
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("EventBus stopped")
    
    def add_handler(self, handler: EventHandler):
        """
        Add event handler.
        
        Args:
            handler: Event handler to add.
        """
        self.handlers.append(handler)
        logger.info(f"Event handler added: {handler.get_handler_name()}")
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """
        Subscribe to event type.
        
        Args:
            event_type: Event type to subscribe to.
            callback: Callback function.
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        logger.info(f"Subscribed to event type: {event_type.value}")
    
    def add_middleware(self, middleware: Callable[[Event], Event]):
        """
        Add middleware for event processing.
        
        Args:
            middleware: Middleware function.
        """
        self.middleware.append(middleware)
    
    def publish(self, event: Event) -> bool:
        """
        Publish event to the bus.
        
        Args:
            event: Event to publish.
            
        Returns:
            True if successful.
        """
        try:
            # Apply middleware
            processed_event = event
            for middleware in self.middleware:
                processed_event = middleware(processed_event)
            
            # Store event
            if self.event_store.append_event(processed_event):
                # Queue for processing
                self.processing_queue.put(processed_event)
                logger.debug(f"Event published: {event.event_type.value}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            return False
    
    def publish_events(self, events: List[Event]) -> bool:
        """
        Publish multiple events.
        
        Args:
            events: List of events to publish.
            
        Returns:
            True if all successful.
        """
        try:
            success_count = 0
            for event in events:
                if self.publish(event):
                    success_count += 1
            
            return success_count == len(events)
            
        except Exception as e:
            logger.error(f"Error publishing events: {e}")
            return False
    
    def _process_events(self):
        """Process events from queue."""
        while self.is_processing:
            try:
                # Get event from queue with timeout
                try:
                    event = self.processing_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process with handlers
                for handler in self.handlers:
                    if handler.can_handle(event):
                        try:
                            # Run handler asynchronously
                            future = self.executor.submit(self._run_handler, handler, event)
                            result_events = future.result(timeout=30)
                            
                            # Publish resulting events
                            if result_events:
                                for result_event in result_events:
                                    self.processing_queue.put(result_event)
                                    
                        except Exception as e:
                            logger.error(f"Error in handler {handler.get_handler_name()}: {e}")
                
                # Call subscribers
                if event.event_type in self.subscribers:
                    for callback in self.subscribers[event.event_type]:
                        try:
                            self.executor.submit(callback, event)
                        except Exception as e:
                            logger.error(f"Error in subscriber callback: {e}")
                
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def _run_handler(self, handler: EventHandler, event: Event) -> Optional[List[Event]]:
        """Run event handler."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(handler.handle(event))
        finally:
            loop.close()


class CommandBus:
    """Command bus for CQRS pattern."""
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize command bus.
        
        Args:
            event_bus: Event bus instance.
        """
        self.event_bus = event_bus
        self.handlers: List[CommandHandler] = []
        self.middleware: List[Callable] = []
        
        logger.info("CommandBus initialized")
    
    def add_handler(self, handler: CommandHandler):
        """
        Add command handler.
        
        Args:
            handler: Command handler to add.
        """
        self.handlers.append(handler)
        logger.info(f"Command handler added: {handler.__class__.__name__}")
    
    def add_middleware(self, middleware: Callable[[Command], Command]):
        """
        Add middleware for command processing.
        
        Args:
            middleware: Middleware function.
        """
        self.middleware.append(middleware)
    
    async def execute(self, command: Command) -> bool:
        """
        Execute command.
        
        Args:
            command: Command to execute.
            
        Returns:
            True if successful.
        """
        try:
            # Apply middleware
            processed_command = command
            for middleware in self.middleware:
                processed_command = middleware(processed_command)
            
            # Find handler
            for handler in self.handlers:
                if handler.can_handle(processed_command):
                    # Execute handler
                    events = await handler.handle(processed_command)
                    
                    # Publish resulting events
                    if events:
                        return self.event_bus.publish_events(events)
                    
                    return True
            
            logger.warning(f"No handler found for command: {command.command_type}")
            return False
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return False


class QueryBus:
    """Query bus for CQRS pattern."""
    
    def __init__(self):
        """Initialize query bus."""
        self.handlers: List[QueryHandler] = []
        self.middleware: List[Callable] = []
        
        logger.info("QueryBus initialized")
    
    def add_handler(self, handler: QueryHandler):
        """
        Add query handler.
        
        Args:
            handler: Query handler to add.
        """
        self.handlers.append(handler)
        logger.info(f"Query handler added: {handler.__class__.__name__}")
    
    def add_middleware(self, middleware: Callable[[Query], Query]):
        """
        Add middleware for query processing.
        
        Args:
            middleware: Middleware function.
        """
        self.middleware.append(middleware)
    
    async def execute(self, query: Query) -> Any:
        """
        Execute query.
        
        Args:
            query: Query to execute.
            
        Returns:
            Query result.
        """
        try:
            # Apply middleware
            processed_query = query
            for middleware in self.middleware:
                processed_query = middleware(processed_query)
            
            # Find handler
            for handler in self.handlers:
                if handler.can_handle(processed_query):
                    return await handler.handle(processed_query)
            
            logger.warning(f"No handler found for query: {query.query_type}")
            return None
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None


class Saga:
    """Saga pattern for distributed transactions."""
    
    def __init__(self, saga_id: str, event_bus: EventBus):
        """
        Initialize saga.
        
        Args:
            saga_id: Unique saga identifier.
            event_bus: Event bus instance.
        """
        self.saga_id = saga_id
        self.event_bus = event_bus
        self.steps: List[Dict[str, Any]] = []
        self.current_step = 0
        self.completed = False
        self.compensated = False
        self.state: Dict[str, Any] = {}
        
        logger.info(f"Saga initialized: {saga_id}")
    
    def add_step(self, step_name: str, execute_command: Command, 
                compensate_command: Optional[Command] = None):
        """
        Add step to saga.
        
        Args:
            step_name: Name of the step.
            execute_command: Command to execute.
            compensate_command: Command to compensate (rollback).
        """
        step = {
            "name": step_name,
            "execute_command": execute_command,
            "compensate_command": compensate_command,
            "completed": False,
            "compensated": False
        }
        self.steps.append(step)
    
    async def execute(self, command_bus: CommandBus) -> bool:
        """
        Execute saga.
        
        Args:
            command_bus: Command bus instance.
            
        Returns:
            True if successful.
        """
        try:
            for i, step in enumerate(self.steps[self.current_step:], self.current_step):
                logger.info(f"Executing saga step: {step['name']}")
                
                success = await command_bus.execute(step["execute_command"])
                
                if success:
                    step["completed"] = True
                    self.current_step = i + 1
                else:
                    # Failure - start compensation
                    await self._compensate(command_bus, i)
                    return False
            
            self.completed = True
            logger.info(f"Saga completed: {self.saga_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing saga {self.saga_id}: {e}")
            await self._compensate(command_bus, self.current_step - 1)
            return False
    
    async def _compensate(self, command_bus: CommandBus, from_step: int):
        """
        Compensate (rollback) saga steps.
        
        Args:
            command_bus: Command bus instance.
            from_step: Step index to start compensation from.
        """
        logger.warning(f"Starting compensation for saga: {self.saga_id}")
        
        # Compensate in reverse order
        for i in range(from_step, -1, -1):
            step = self.steps[i]
            
            if step["completed"] and not step["compensated"] and step["compensate_command"]:
                logger.info(f"Compensating saga step: {step['name']}")
                
                try:
                    await command_bus.execute(step["compensate_command"])
                    step["compensated"] = True
                except Exception as e:
                    logger.error(f"Error compensating step {step['name']}: {e}")
        
        self.compensated = True
        logger.info(f"Saga compensation completed: {self.saga_id}")


# Event projection for read models
class EventProjection:
    """Event projection for building read models."""
    
    def __init__(self, projection_name: str, event_store: EventStore):
        """
        Initialize event projection.
        
        Args:
            projection_name: Name of the projection.
            event_store: Event store instance.
        """
        self.projection_name = projection_name
        self.event_store = event_store
        self.read_model: Dict[str, Any] = {}
        self.last_processed_sequence = 0
        
        logger.info(f"EventProjection initialized: {projection_name}")
    
    def project_event(self, event: Event):
        """
        Project event into read model.
        
        Args:
            event: Event to project.
        """
        # Override in subclasses for specific projections
        pass
    
    def rebuild(self):
        """Rebuild projection from all events."""
        self.read_model = {}
        self.last_processed_sequence = 0
        
        events = self.event_store.get_events()
        for event in events:
            if event.sequence_number and event.sequence_number > self.last_processed_sequence:
                self.project_event(event)
                self.last_processed_sequence = event.sequence_number
        
        logger.info(f"Projection rebuilt: {self.projection_name}")
    
    def update(self):
        """Update projection with new events."""
        events = self.event_store.get_events(from_sequence=self.last_processed_sequence + 1)
        
        for event in events:
            if event.sequence_number and event.sequence_number > self.last_processed_sequence:
                self.project_event(event)
                self.last_processed_sequence = event.sequence_number
    
    def get_read_model(self) -> Dict[str, Any]:
        """Get current read model."""
        return self.read_model.copy()


# Global instances
_event_store = None
_event_bus = None
_command_bus = None
_query_bus = None


def get_event_store() -> EventStore:
    """Get global event store."""
    global _event_store
    if _event_store is None:
        _event_store = EventStore()
    return _event_store


def get_event_bus() -> EventBus:
    """Get global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus(get_event_store())
    return _event_bus


def get_command_bus() -> CommandBus:
    """Get global command bus."""
    global _command_bus
    if _command_bus is None:
        _command_bus = CommandBus(get_event_bus())
    return _command_bus


def get_query_bus() -> QueryBus:
    """Get global query bus."""
    global _query_bus
    if _query_bus is None:
        _query_bus = QueryBus()
    return _query_bus


def start_event_system():
    """Start the event-driven system."""
    event_bus = get_event_bus()
    event_bus.start()
    logger.info("Event-driven system started")


def stop_event_system():
    """Stop the event-driven system."""
    event_bus = get_event_bus()
    event_bus.stop()
    logger.info("Event-driven system stopped")


# Convenience functions for common events
def publish_system_event(event_type: EventType, data: Dict[str, Any] = None):
    """Publish a system event."""
    event = Event(
        event_type=event_type,
        source="system",
        data=data or {}
    )
    get_event_bus().publish(event)


def publish_fusion_event(event_type: EventType, aggregate_id: str, data: Dict[str, Any] = None):
    """Publish a fusion-related event."""
    event = Event(
        event_type=event_type,
        source="fusion_analyzer",
        aggregate_id=aggregate_id,
        aggregate_type="fusion_analysis",
        data=data or {}
    )
    get_event_bus().publish(event)


def publish_user_event(event_type: EventType, user_id: str, data: Dict[str, Any] = None):
    """Publish a user-related event."""
    event = Event(
        event_type=event_type,
        source="user_service",
        aggregate_id=user_id,
        aggregate_type="user",
        data=data or {}
    )
    get_event_bus().publish(event)


# Import messaging and configuration components
from .messaging import (
    Message, MessageQueue, MessageBroker, EventStream,
    MessageStatus, MessagePriority, get_message_broker, 
    get_event_stream, start_messaging, stop_messaging
)

from .config import (
    EventSchema, RoutingRule, RetentionConfig, EventConfiguration,
    EventSchemaType, RetentionPolicy, SecurityLevel, get_event_config
)

# Global instances for easy access
event_store = get_event_store()
event_bus = get_event_bus()
command_bus = get_command_bus()
query_bus = get_query_bus()
message_broker = get_message_broker()
event_config = get_event_config()

# Initialize with default handlers
from .handlers import (
    PlasmaMonitoringHandler,
    SafetySystemHandler,
    MLModelHandler,
    AnalysisWorkflowHandler
)

# Register default handlers
plasma_handler = PlasmaMonitoringHandler()
safety_handler = SafetySystemHandler()
ml_handler = MLModelHandler()
workflow_handler = AnalysisWorkflowHandler()

# Register event handlers
event_bus.subscribe(EventType.PLASMA_TEMPERATURE_CHANGED, plasma_handler.handle_plasma_event)
event_bus.subscribe(EventType.PLASMA_DENSITY_CHANGED, plasma_handler.handle_plasma_event)
event_bus.subscribe(EventType.PLASMA_INSTABILITY_DETECTED, plasma_handler.handle_plasma_event)
event_bus.subscribe(EventType.SAFETY_THRESHOLD_EXCEEDED, safety_handler.handle_safety_event)
event_bus.subscribe(EventType.EMERGENCY_SHUTDOWN, safety_handler.handle_safety_event)
event_bus.subscribe(EventType.MODEL_TRAINED, ml_handler.handle_ml_event)
event_bus.subscribe(EventType.MODEL_DEPLOYED, ml_handler.handle_ml_event)
event_bus.subscribe(EventType.ANALYSIS_STARTED, workflow_handler.handle_analysis_event)
event_bus.subscribe(EventType.ANALYSIS_COMPLETED, workflow_handler.handle_analysis_event)

# Set up default message broker queues and topics
message_broker.create_queue("fusion_events", max_size=5000)
message_broker.create_queue("critical_events", max_size=1000)
message_broker.create_queue("safety_events", max_size=2000)
message_broker.create_queue("ml_events", max_size=3000)

# Bind queues to topics
message_broker.bind_queue("fusion_events", "fusion.*")
message_broker.bind_queue("critical_events", "*.critical")
message_broker.bind_queue("safety_events", "safety.*")
message_broker.bind_queue("ml_events", "ml.*")

# Add routing rules
message_broker.add_routing_rule("fusion.analysis.*", "fusion_events")
message_broker.add_routing_rule("safety.*", "safety_events")
message_broker.add_routing_rule("ml.model.*", "ml_events")
message_broker.add_routing_rule("*.critical", "critical_events")

# Start messaging system
start_messaging()

# Start event system
start_event_system()