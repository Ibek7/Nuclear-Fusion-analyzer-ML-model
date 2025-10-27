"""
Event Configuration System.

This module provides:
- Event schema definitions and validation
- Event routing configuration
- Event transformation rules
- Event archival and retention policies
- Event security and access control
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import os
from pathlib import Path

from . import EventType

logger = logging.getLogger(__name__)


class EventSchemaType(Enum):
    """Event schema types."""
    JSON_SCHEMA = "json_schema"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    CUSTOM = "custom"


class RetentionPolicy(Enum):
    """Event retention policies."""
    TIME_BASED = "time_based"
    SIZE_BASED = "size_based"
    COUNT_BASED = "count_based"
    CUSTOM = "custom"


class SecurityLevel(Enum):
    """Event security levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class EventSchema:
    """Event schema definition."""
    name: str
    version: str
    event_type: EventType
    schema_type: EventSchemaType = EventSchemaType.JSON_SCHEMA
    schema_definition: Dict[str, Any] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    created_at: datetime = field(default_factory=datetime.now)
    
    def validate_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Validate event data against schema.
        
        Args:
            event_data: Event data to validate.
            
        Returns:
            True if valid.
        """
        try:
            # Check required fields
            for field_name in self.required_fields:
                if field_name not in event_data:
                    logger.error(f"Missing required field: {field_name}")
                    return False
            
            # Apply validation rules
            for field_name, rules in self.validation_rules.items():
                if field_name in event_data:
                    value = event_data[field_name]
                    
                    if not self._validate_field(value, rules):
                        logger.error(f"Validation failed for field: {field_name}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False
    
    def transform_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform event data using configured transformations.
        
        Args:
            event_data: Event data to transform.
            
        Returns:
            Transformed event data.
        """
        try:
            result = event_data.copy()
            
            for transformation in self.transformations:
                transform_type = transformation.get("type")
                
                if transform_type == "rename_field":
                    old_name = transformation["from"]
                    new_name = transformation["to"]
                    if old_name in result:
                        result[new_name] = result.pop(old_name)
                
                elif transform_type == "add_field":
                    field_name = transformation["field"]
                    field_value = transformation["value"]
                    result[field_name] = field_value
                
                elif transform_type == "remove_field":
                    field_name = transformation["field"]
                    result.pop(field_name, None)
                
                elif transform_type == "format_field":
                    field_name = transformation["field"]
                    format_string = transformation["format"]
                    if field_name in result:
                        result[field_name] = format_string.format(result[field_name])
            
            return result
            
        except Exception as e:
            logger.error(f"Event transformation error: {e}")
            return event_data
    
    def _validate_field(self, value: Any, rules: Dict[str, Any]) -> bool:
        """
        Validate field value against rules.
        
        Args:
            value: Field value.
            rules: Validation rules.
            
        Returns:
            True if valid.
        """
        # Type validation
        if "type" in rules:
            expected_type = rules["type"]
            if expected_type == "string" and not isinstance(value, str):
                return False
            elif expected_type == "number" and not isinstance(value, (int, float)):
                return False
            elif expected_type == "boolean" and not isinstance(value, bool):
                return False
            elif expected_type == "array" and not isinstance(value, list):
                return False
            elif expected_type == "object" and not isinstance(value, dict):
                return False
        
        # Range validation
        if "min" in rules and value < rules["min"]:
            return False
        if "max" in rules and value > rules["max"]:
            return False
        
        # Length validation
        if "min_length" in rules and len(str(value)) < rules["min_length"]:
            return False
        if "max_length" in rules and len(str(value)) > rules["max_length"]:
            return False
        
        # Pattern validation
        if "pattern" in rules:
            import re
            if not re.match(rules["pattern"], str(value)):
                return False
        
        # Enum validation
        if "enum" in rules and value not in rules["enum"]:
            return False
        
        return True


@dataclass
class RoutingRule:
    """Event routing rule."""
    name: str
    condition: str  # Expression to evaluate
    destinations: List[str] = field(default_factory=list)
    priority: int = 0
    enabled: bool = True
    filters: Dict[str, Any] = field(default_factory=dict)
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    
    def matches(self, event_data: Dict[str, Any]) -> bool:
        """
        Check if event matches routing rule.
        
        Args:
            event_data: Event data to check.
            
        Returns:
            True if matches.
        """
        if not self.enabled:
            return False
        
        try:
            # Simple condition evaluation
            # In a real implementation, use a proper expression evaluator
            return self._evaluate_condition(self.condition, event_data)
            
        except Exception as e:
            logger.error(f"Error evaluating routing condition: {e}")
            return False
    
    def _evaluate_condition(self, condition: str, event_data: Dict[str, Any]) -> bool:
        """
        Evaluate routing condition.
        
        Args:
            condition: Condition string.
            event_data: Event data.
            
        Returns:
            True if condition is met.
        """
        # Simple condition evaluation (extend as needed)
        if "event_type ==" in condition:
            _, value = condition.split("event_type ==")
            target_type = value.strip().strip("'\"")
            return event_data.get("event_type") == target_type
        
        if "priority >=" in condition:
            _, value = condition.split("priority >=")
            target_priority = int(value.strip())
            return event_data.get("priority", 0) >= target_priority
        
        if "contains" in condition:
            field, value = condition.split("contains")
            field_name = field.strip()
            search_value = value.strip().strip("'\"")
            field_value = str(event_data.get(field_name, ""))
            return search_value in field_value
        
        return True


@dataclass
class RetentionConfig:
    """Event retention configuration."""
    policy: RetentionPolicy
    max_age: Optional[timedelta] = None
    max_size_mb: Optional[int] = None
    max_count: Optional[int] = None
    archive_enabled: bool = False
    archive_location: str = ""
    compression_enabled: bool = True
    encryption_enabled: bool = False
    
    def should_retain(self, event_age: timedelta, total_size_mb: int, total_count: int) -> bool:
        """
        Check if event should be retained.
        
        Args:
            event_age: Age of the event.
            total_size_mb: Total size in MB.
            total_count: Total event count.
            
        Returns:
            True if should retain.
        """
        if self.policy == RetentionPolicy.TIME_BASED and self.max_age:
            return event_age <= self.max_age
        
        if self.policy == RetentionPolicy.SIZE_BASED and self.max_size_mb:
            return total_size_mb <= self.max_size_mb
        
        if self.policy == RetentionPolicy.COUNT_BASED and self.max_count:
            return total_count <= self.max_count
        
        return True


class EventConfiguration:
    """Event system configuration manager."""
    
    def __init__(self, config_path: str = "config/events.yaml"):
        """
        Initialize event configuration.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config_path = config_path
        self.schemas: Dict[str, EventSchema] = {}
        self.routing_rules: List[RoutingRule] = []
        self.retention_configs: Dict[EventType, RetentionConfig] = {}
        self.global_config: Dict[str, Any] = {}
        
        self._load_configuration()
        logger.info("EventConfiguration initialized")
    
    def _load_configuration(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                self._parse_configuration(config_data)
                logger.info(f"Configuration loaded from: {self.config_path}")
            else:
                self._create_default_configuration()
                logger.info("Default configuration created")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_configuration()
    
    def _parse_configuration(self, config_data: Dict[str, Any]):
        """
        Parse configuration data.
        
        Args:
            config_data: Configuration data.
        """
        # Parse global configuration
        self.global_config = config_data.get("global", {})
        
        # Parse schemas
        schemas_data = config_data.get("schemas", {})
        for schema_name, schema_config in schemas_data.items():
            schema = EventSchema(
                name=schema_name,
                version=schema_config.get("version", "1.0"),
                event_type=EventType(schema_config.get("event_type", "SYSTEM")),
                schema_type=EventSchemaType(schema_config.get("schema_type", "json_schema")),
                schema_definition=schema_config.get("schema_definition", {}),
                required_fields=schema_config.get("required_fields", []),
                optional_fields=schema_config.get("optional_fields", []),
                validation_rules=schema_config.get("validation_rules", {}),
                transformations=schema_config.get("transformations", []),
                security_level=SecurityLevel(schema_config.get("security_level", "internal"))
            )
            self.schemas[schema_name] = schema
        
        # Parse routing rules
        routing_data = config_data.get("routing", [])
        for rule_config in routing_data:
            rule = RoutingRule(
                name=rule_config.get("name", ""),
                condition=rule_config.get("condition", ""),
                destinations=rule_config.get("destinations", []),
                priority=rule_config.get("priority", 0),
                enabled=rule_config.get("enabled", True),
                filters=rule_config.get("filters", {}),
                transformations=rule_config.get("transformations", [])
            )
            self.routing_rules.append(rule)
        
        # Parse retention configurations
        retention_data = config_data.get("retention", {})
        for event_type_str, retention_config in retention_data.items():
            event_type = EventType(event_type_str)
            
            max_age = None
            if "max_age_days" in retention_config:
                max_age = timedelta(days=retention_config["max_age_days"])
            
            retention = RetentionConfig(
                policy=RetentionPolicy(retention_config.get("policy", "time_based")),
                max_age=max_age,
                max_size_mb=retention_config.get("max_size_mb"),
                max_count=retention_config.get("max_count"),
                archive_enabled=retention_config.get("archive_enabled", False),
                archive_location=retention_config.get("archive_location", ""),
                compression_enabled=retention_config.get("compression_enabled", True),
                encryption_enabled=retention_config.get("encryption_enabled", False)
            )
            self.retention_configs[event_type] = retention
    
    def _create_default_configuration(self):
        """Create default configuration."""
        # Default schemas
        fusion_analysis_schema = EventSchema(
            name="fusion_analysis",
            version="1.0",
            event_type=EventType.FUSION_ANALYSIS,
            required_fields=["temperature", "density", "timestamp"],
            optional_fields=["pressure", "magnetic_field"],
            validation_rules={
                "temperature": {"type": "number", "min": 0, "max": 100000000},
                "density": {"type": "number", "min": 0},
                "timestamp": {"type": "string", "pattern": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"}
            }
        )
        self.schemas["fusion_analysis"] = fusion_analysis_schema
        
        # Default routing rules
        critical_rule = RoutingRule(
            name="critical_events",
            condition="priority >= 4",
            destinations=["critical_queue", "alert_system"],
            priority=1
        )
        self.routing_rules.append(critical_rule)
        
        # Default retention policies
        self.retention_configs[EventType.SYSTEM] = RetentionConfig(
            policy=RetentionPolicy.TIME_BASED,
            max_age=timedelta(days=30)
        )
        
        self.retention_configs[EventType.FUSION_ANALYSIS] = RetentionConfig(
            policy=RetentionPolicy.TIME_BASED,
            max_age=timedelta(days=365),
            archive_enabled=True
        )
    
    def get_schema(self, name: str) -> Optional[EventSchema]:
        """
        Get event schema by name.
        
        Args:
            name: Schema name.
            
        Returns:
            Event schema or None.
        """
        return self.schemas.get(name)
    
    def add_schema(self, schema: EventSchema):
        """
        Add event schema.
        
        Args:
            schema: Event schema to add.
        """
        self.schemas[schema.name] = schema
        logger.info(f"Schema added: {schema.name}")
    
    def get_routing_rules(self, event_data: Dict[str, Any]) -> List[RoutingRule]:
        """
        Get matching routing rules for event.
        
        Args:
            event_data: Event data.
            
        Returns:
            List of matching routing rules.
        """
        matching_rules = []
        
        for rule in self.routing_rules:
            if rule.matches(event_data):
                matching_rules.append(rule)
        
        # Sort by priority
        matching_rules.sort(key=lambda r: r.priority, reverse=True)
        return matching_rules
    
    def get_retention_config(self, event_type: EventType) -> RetentionConfig:
        """
        Get retention configuration for event type.
        
        Args:
            event_type: Event type.
            
        Returns:
            Retention configuration.
        """
        return self.retention_configs.get(
            event_type,
            RetentionConfig(policy=RetentionPolicy.TIME_BASED, max_age=timedelta(days=30))
        )
    
    def save_configuration(self):
        """Save configuration to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            config_data = {
                "global": self.global_config,
                "schemas": {
                    name: {
                        "version": schema.version,
                        "event_type": schema.event_type.value,
                        "schema_type": schema.schema_type.value,
                        "schema_definition": schema.schema_definition,
                        "required_fields": schema.required_fields,
                        "optional_fields": schema.optional_fields,
                        "validation_rules": schema.validation_rules,
                        "transformations": schema.transformations,
                        "security_level": schema.security_level.value
                    }
                    for name, schema in self.schemas.items()
                },
                "routing": [
                    {
                        "name": rule.name,
                        "condition": rule.condition,
                        "destinations": rule.destinations,
                        "priority": rule.priority,
                        "enabled": rule.enabled,
                        "filters": rule.filters,
                        "transformations": rule.transformations
                    }
                    for rule in self.routing_rules
                ],
                "retention": {
                    event_type.value: {
                        "policy": config.policy.value,
                        "max_age_days": config.max_age.days if config.max_age else None,
                        "max_size_mb": config.max_size_mb,
                        "max_count": config.max_count,
                        "archive_enabled": config.archive_enabled,
                        "archive_location": config.archive_location,
                        "compression_enabled": config.compression_enabled,
                        "encryption_enabled": config.encryption_enabled
                    }
                    for event_type, config in self.retention_configs.items()
                }
            }
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def validate_event(self, event_type: EventType, event_data: Dict[str, Any]) -> bool:
        """
        Validate event data.
        
        Args:
            event_type: Event type.
            event_data: Event data.
            
        Returns:
            True if valid.
        """
        # Find schema for event type
        for schema in self.schemas.values():
            if schema.event_type == event_type:
                return schema.validate_event(event_data)
        
        # No schema found, basic validation
        return True
    
    def transform_event(self, event_type: EventType, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform event data.
        
        Args:
            event_type: Event type.
            event_data: Event data.
            
        Returns:
            Transformed event data.
        """
        # Find schema for event type
        for schema in self.schemas.values():
            if schema.event_type == event_type:
                return schema.transform_event(event_data)
        
        # No schema found, return as-is
        return event_data


# Global configuration instance
_event_config = None


def get_event_config() -> EventConfiguration:
    """Get global event configuration."""
    global _event_config
    if _event_config is None:
        _event_config = EventConfiguration()
    return _event_config