"""
Database query utilities and repositories.

This module provides high-level query interfaces, repository patterns,
and data access utilities for fusion analysis data.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncGenerator
from dataclasses import dataclass
import uuid
from pathlib import Path
import json

import pandas as pd
import numpy as np

from .connection import get_connection_manager
from .models import (
    FusionExperiment, FusionShot, TimeSeriesData, ModelPrediction,
    ExperimentStatus, ShotStatus, QualityFlag
)

logger = logging.getLogger(__name__)


@dataclass
class QueryFilter:
    """Generic query filter container."""
    
    field: str
    operator: str  # eq, ne, gt, gte, lt, lte, in, not_in, like, between
    value: Any
    
    def to_sql_condition(self, table_alias: str = "") -> str:
        """Convert filter to SQL condition."""
        field_name = f"{table_alias}.{self.field}" if table_alias else self.field
        
        if self.operator == "eq":
            return f"{field_name} = :value"
        elif self.operator == "ne":
            return f"{field_name} != :value"
        elif self.operator == "gt":
            return f"{field_name} > :value"
        elif self.operator == "gte":
            return f"{field_name} >= :value"
        elif self.operator == "lt":
            return f"{field_name} < :value"
        elif self.operator == "lte":
            return f"{field_name} <= :value"
        elif self.operator == "in":
            return f"{field_name} IN :value"
        elif self.operator == "not_in":
            return f"{field_name} NOT IN :value"
        elif self.operator == "like":
            return f"{field_name} ILIKE :value"
        elif self.operator == "between":
            return f"{field_name} BETWEEN :value_start AND :value_end"
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")


@dataclass 
class QueryOptions:
    """Query options for pagination and sorting."""
    
    limit: int = 100
    offset: int = 0
    order_by: str = "created_at"
    order_direction: str = "desc"  # asc or desc
    include_related: bool = False


class ExperimentRepository:
    """
    Repository for fusion experiment data operations.
    
    Provides high-level interfaces for CRUD operations,
    complex queries, and data aggregations.
    """
    
    def __init__(self):
        """Initialize experiment repository."""
        self.connection_manager = get_connection_manager()
        
        logger.info("ExperimentRepository initialized")
    
    async def create_experiment(self, 
                               name: str,
                               reactor_type: str,
                               experiment_type: str = "simulation",
                               **kwargs) -> FusionExperiment:
        """
        Create a new fusion experiment.
        
        Args:
            name: Experiment name.
            reactor_type: Type of reactor.
            experiment_type: Type of experiment.
            **kwargs: Additional experiment parameters.
            
        Returns:
            Created experiment object.
        """
        try:
            # Import here to avoid circular imports
            import sqlalchemy as sa
            from sqlalchemy.orm import sessionmaker
            
            async with self.connection_manager.get_postgresql_connection() as conn:
                # Create experiment record
                experiment_data = {
                    'id': uuid.uuid4(),
                    'name': name,
                    'reactor_type': reactor_type,
                    'experiment_type': experiment_type,
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc),
                    'status': ExperimentStatus.PLANNED.value,
                    **kwargs
                }
                
                # Insert experiment
                query = """
                INSERT INTO fusion_experiments 
                (id, name, reactor_type, experiment_type, created_at, updated_at, status, description, major_radius, minor_radius, magnetic_field, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING *
                """
                
                result = await conn.fetchrow(
                    query,
                    experiment_data['id'],
                    experiment_data['name'],
                    experiment_data['reactor_type'],
                    experiment_data['experiment_type'],
                    experiment_data['created_at'],
                    experiment_data['updated_at'],
                    experiment_data['status'],
                    kwargs.get('description'),
                    kwargs.get('major_radius'),
                    kwargs.get('minor_radius'),
                    kwargs.get('magnetic_field'),
                    json.dumps(kwargs.get('metadata', {}))
                )
                
                logger.info(f"Created experiment: {name} ({experiment_data['id']})")
                
                # Convert result to FusionExperiment-like object
                experiment = type('FusionExperiment', (), dict(result))()
                return experiment
                
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    async def get_experiment_by_id(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment by ID.
        
        Args:
            experiment_id: Experiment ID.
            
        Returns:
            Experiment data or None.
        """
        try:
            async with self.connection_manager.get_postgresql_connection() as conn:
                query = """
                SELECT * FROM fusion_experiments WHERE id = $1
                """
                
                result = await conn.fetchrow(query, uuid.UUID(experiment_id))
                
                if result:
                    return dict(result)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get experiment {experiment_id}: {e}")
            return None
    
    async def list_experiments(self, 
                              filters: List[QueryFilter] = None,
                              options: QueryOptions = None) -> List[Dict[str, Any]]:
        """
        List experiments with filtering and pagination.
        
        Args:
            filters: Query filters to apply.
            options: Query options for pagination and sorting.
            
        Returns:
            List of experiment records.
        """
        if options is None:
            options = QueryOptions()
        
        try:
            async with self.connection_manager.get_postgresql_connection() as conn:
                # Build query
                query = "SELECT * FROM fusion_experiments"
                params = []
                param_count = 0
                
                # Add filters
                if filters:
                    conditions = []
                    for filter_obj in filters:
                        param_count += 1
                        if filter_obj.operator == "between":
                            conditions.append(f"{filter_obj.field} BETWEEN ${param_count} AND ${param_count + 1}")
                            params.extend([filter_obj.value[0], filter_obj.value[1]])
                            param_count += 1
                        else:
                            conditions.append(f"{filter_obj.field} = ${param_count}")
                            params.append(filter_obj.value)
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                # Add ordering
                query += f" ORDER BY {options.order_by} {options.order_direction.upper()}"
                
                # Add pagination
                param_count += 1
                query += f" LIMIT ${param_count}"
                params.append(options.limit)
                
                param_count += 1
                query += f" OFFSET ${param_count}"
                params.append(options.offset)
                
                # Execute query
                results = await conn.fetch(query, *params)
                
                return [dict(result) for result in results]
                
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    async def update_experiment(self, 
                               experiment_id: str,
                               updates: Dict[str, Any]) -> bool:
        """
        Update experiment data.
        
        Args:
            experiment_id: Experiment ID.
            updates: Fields to update.
            
        Returns:
            Success status.
        """
        try:
            async with self.connection_manager.get_postgresql_connection() as conn:
                # Build update query
                set_clauses = []
                params = []
                param_count = 0
                
                for field, value in updates.items():
                    param_count += 1
                    set_clauses.append(f"{field} = ${param_count}")
                    params.append(value)
                
                # Add updated_at
                param_count += 1
                set_clauses.append(f"updated_at = ${param_count}")
                params.append(datetime.now(timezone.utc))
                
                # Add experiment ID
                param_count += 1
                params.append(uuid.UUID(experiment_id))
                
                query = f"""
                UPDATE fusion_experiments 
                SET {', '.join(set_clauses)}
                WHERE id = ${param_count}
                """
                
                result = await conn.execute(query, *params)
                
                logger.info(f"Updated experiment: {experiment_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update experiment {experiment_id}: {e}")
            return False
    
    async def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete experiment and all related data.
        
        Args:
            experiment_id: Experiment ID.
            
        Returns:
            Success status.
        """
        try:
            async with self.connection_manager.get_postgresql_connection() as conn:
                # Delete experiment (cascades to shots and time series)
                query = "DELETE FROM fusion_experiments WHERE id = $1"
                
                result = await conn.execute(query, uuid.UUID(experiment_id))
                
                logger.info(f"Deleted experiment: {experiment_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            return False
    
    async def get_experiment_stats(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment statistics.
        
        Args:
            experiment_id: Experiment ID.
            
        Returns:
            Statistics dictionary.
        """
        try:
            async with self.connection_manager.get_postgresql_connection() as conn:
                query = """
                SELECT 
                    e.name,
                    e.status,
                    e.created_at,
                    COUNT(s.id) as shot_count,
                    COUNT(ts.id) as time_series_count,
                    AVG(s.q_factor) as avg_q_factor,
                    AVG(s.confinement_time) as avg_confinement_time,
                    COUNT(CASE WHEN s.had_disruption THEN 1 END) as disruption_count
                FROM fusion_experiments e
                LEFT JOIN fusion_shots s ON e.id = s.experiment_id
                LEFT JOIN time_series_data ts ON s.id = ts.shot_id
                WHERE e.id = $1
                GROUP BY e.id, e.name, e.status, e.created_at
                """
                
                result = await conn.fetchrow(query, uuid.UUID(experiment_id))
                
                if result:
                    return dict(result)
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get experiment stats {experiment_id}: {e}")
            return {}


class ShotRepository:
    """
    Repository for fusion shot data operations.
    
    Handles shot CRUD operations, time series data,
    and shot-level analytics.
    """
    
    def __init__(self):
        """Initialize shot repository."""
        self.connection_manager = get_connection_manager()
        
        logger.info("ShotRepository initialized")
    
    async def create_shot(self, 
                         experiment_id: str,
                         shot_number: int,
                         plasma_parameters: Dict[str, float],
                         start_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Create a new fusion shot.
        
        Args:
            experiment_id: Parent experiment ID.
            shot_number: Shot number.
            plasma_parameters: Plasma parameter values.
            start_time: Shot start time.
            
        Returns:
            Created shot data.
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)
        
        try:
            async with self.connection_manager.get_postgresql_connection() as conn:
                shot_data = {
                    'id': uuid.uuid4(),
                    'experiment_id': uuid.UUID(experiment_id),
                    'shot_number': shot_number,
                    'start_time': start_time,
                    'status': ShotStatus.COMPLETED.value,
                    **plasma_parameters
                }
                
                query = """
                INSERT INTO fusion_shots 
                (id, experiment_id, shot_number, start_time, status, plasma_current, 
                 electron_density, electron_temperature, ion_temperature, 
                 neutral_beam_power, rf_heating_power)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING *
                """
                
                result = await conn.fetchrow(
                    query,
                    shot_data['id'],
                    shot_data['experiment_id'],
                    shot_data['shot_number'],
                    shot_data['start_time'],
                    shot_data['status'],
                    plasma_parameters.get('plasma_current'),
                    plasma_parameters.get('electron_density'),
                    plasma_parameters.get('electron_temperature'),
                    plasma_parameters.get('ion_temperature'),
                    plasma_parameters.get('neutral_beam_power'),
                    plasma_parameters.get('rf_heating_power')
                )
                
                logger.info(f"Created shot: {shot_number} for experiment {experiment_id}")
                return dict(result)
                
        except Exception as e:
            logger.error(f"Failed to create shot: {e}")
            raise
    
    async def get_shots_for_experiment(self, 
                                      experiment_id: str,
                                      options: QueryOptions = None) -> List[Dict[str, Any]]:
        """
        Get all shots for an experiment.
        
        Args:
            experiment_id: Experiment ID.
            options: Query options.
            
        Returns:
            List of shot records.
        """
        if options is None:
            options = QueryOptions()
        
        try:
            async with self.connection_manager.get_postgresql_connection() as conn:
                query = f"""
                SELECT * FROM fusion_shots 
                WHERE experiment_id = $1 
                ORDER BY {options.order_by} {options.order_direction.upper()}
                LIMIT $2 OFFSET $3
                """
                
                results = await conn.fetch(
                    query, 
                    uuid.UUID(experiment_id),
                    options.limit,
                    options.offset
                )
                
                return [dict(result) for result in results]
                
        except Exception as e:
            logger.error(f"Failed to get shots for experiment {experiment_id}: {e}")
            return []
    
    async def store_time_series_data(self, 
                                    shot_id: str,
                                    data: pd.DataFrame) -> int:
        """
        Store time series data for a shot.
        
        Args:
            shot_id: Shot ID.
            data: Time series data DataFrame.
            
        Returns:
            Number of records stored.
        """
        try:
            async with self.connection_manager.get_postgresql_connection() as conn:
                records_stored = 0
                
                # Prepare data for insertion
                for _, row in data.iterrows():
                    query = """
                    INSERT INTO time_series_data 
                    (id, shot_id, timestamp, time_relative, diagnostic_name, 
                     measurement_type, value, units, uncertainty, quality_flag, 
                     spatial_location)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """
                    
                    await conn.execute(
                        query,
                        uuid.uuid4(),
                        uuid.UUID(shot_id),
                        row.get('timestamp', datetime.now(timezone.utc)),
                        row['time_relative'],
                        row['diagnostic_name'],
                        row['measurement_type'],
                        row['value'],
                        row['units'],
                        row.get('uncertainty'),
                        row.get('quality_flag', QualityFlag.GOOD.value),
                        row.get('spatial_location')
                    )
                    
                    records_stored += 1
                
                logger.info(f"Stored {records_stored} time series records for shot {shot_id}")
                return records_stored
                
        except Exception as e:
            logger.error(f"Failed to store time series data: {e}")
            return 0
    
    async def get_time_series_data(self, 
                                  shot_id: str,
                                  diagnostic_name: Optional[str] = None,
                                  time_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Get time series data for a shot.
        
        Args:
            shot_id: Shot ID.
            diagnostic_name: Optional diagnostic filter.
            time_range: Optional time range filter (start, end).
            
        Returns:
            Time series data as DataFrame.
        """
        try:
            async with self.connection_manager.get_postgresql_connection() as conn:
                query = "SELECT * FROM time_series_data WHERE shot_id = $1"
                params = [uuid.UUID(shot_id)]
                param_count = 1
                
                if diagnostic_name:
                    param_count += 1
                    query += f" AND diagnostic_name = ${param_count}"
                    params.append(diagnostic_name)
                
                if time_range:
                    param_count += 1
                    query += f" AND time_relative >= ${param_count}"
                    params.append(time_range[0])
                    
                    param_count += 1
                    query += f" AND time_relative <= ${param_count}"
                    params.append(time_range[1])
                
                query += " ORDER BY time_relative"
                
                results = await conn.fetch(query, *params)
                
                # Convert to DataFrame
                if results:
                    data = [dict(result) for result in results]
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to get time series data: {e}")
            return pd.DataFrame()


class AnalyticsRepository:
    """
    Repository for analytics and aggregated data operations.
    
    Provides complex queries, statistical analysis,
    and performance metrics.
    """
    
    def __init__(self):
        """Initialize analytics repository."""
        self.connection_manager = get_connection_manager()
        
        logger.info("AnalyticsRepository initialized")
    
    async def get_experiment_performance_metrics(self, 
                                               experiment_id: str) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for an experiment.
        
        Args:
            experiment_id: Experiment ID.
            
        Returns:
            Performance metrics dictionary.
        """
        try:
            async with self.connection_manager.get_postgresql_connection() as conn:
                query = """
                SELECT 
                    COUNT(s.id) as total_shots,
                    AVG(s.q_factor) as avg_q_factor,
                    STDDEV(s.q_factor) as std_q_factor,
                    MAX(s.q_factor) as max_q_factor,
                    MIN(s.q_factor) as min_q_factor,
                    AVG(s.confinement_time) as avg_confinement_time,
                    AVG(s.beta_normalized) as avg_beta_normalized,
                    COUNT(CASE WHEN s.had_disruption THEN 1 END) as disruption_count,
                    COUNT(CASE WHEN s.had_disruption THEN 1 END)::float / COUNT(s.id) as disruption_rate,
                    AVG(s.duration) as avg_duration,
                    COUNT(CASE WHEN s.quality_rating >= 0.8 THEN 1 END) as high_quality_shots
                FROM fusion_shots s
                WHERE s.experiment_id = $1
                """
                
                result = await conn.fetchrow(query, uuid.UUID(experiment_id))
                
                if result:
                    return dict(result)
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    async def get_diagnostic_statistics(self, 
                                       experiment_id: str,
                                       diagnostic_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific diagnostic.
        
        Args:
            experiment_id: Experiment ID.
            diagnostic_name: Diagnostic name.
            
        Returns:
            Diagnostic statistics.
        """
        try:
            async with self.connection_manager.get_postgresql_connection() as conn:
                query = """
                SELECT 
                    ts.diagnostic_name,
                    ts.measurement_type,
                    COUNT(ts.id) as data_points,
                    AVG(ts.value) as mean_value,
                    STDDEV(ts.value) as std_value,
                    MIN(ts.value) as min_value,
                    MAX(ts.value) as max_value,
                    COUNT(CASE WHEN ts.quality_flag = 0 THEN 1 END) as good_quality_points,
                    COUNT(CASE WHEN ts.quality_flag = 1 THEN 1 END) as suspect_quality_points,
                    COUNT(CASE WHEN ts.quality_flag = 2 THEN 1 END) as bad_quality_points
                FROM time_series_data ts
                JOIN fusion_shots s ON ts.shot_id = s.id
                WHERE s.experiment_id = $1 AND ts.diagnostic_name = $2
                GROUP BY ts.diagnostic_name, ts.measurement_type
                """
                
                results = await conn.fetch(
                    query, 
                    uuid.UUID(experiment_id),
                    diagnostic_name
                )
                
                return [dict(result) for result in results]
                
        except Exception as e:
            logger.error(f"Failed to get diagnostic statistics: {e}")
            return []
    
    async def get_trending_analysis(self, 
                                   experiment_id: str,
                                   parameter: str,
                                   time_window_days: int = 30) -> Dict[str, Any]:
        """
        Get trending analysis for a parameter.
        
        Args:
            experiment_id: Experiment ID.
            parameter: Parameter to analyze.
            time_window_days: Time window for analysis.
            
        Returns:
            Trending analysis results.
        """
        try:
            async with self.connection_manager.get_postgresql_connection() as conn:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=time_window_days)
                
                # Get parameter column from table structure
                parameter_column = f"s.{parameter}"  # Assuming parameter is in shots table
                
                query = f"""
                SELECT 
                    DATE_TRUNC('day', s.start_time) as date,
                    COUNT(s.id) as shot_count,
                    AVG({parameter_column}) as avg_value,
                    STDDEV({parameter_column}) as std_value,
                    MIN({parameter_column}) as min_value,
                    MAX({parameter_column}) as max_value
                FROM fusion_shots s
                WHERE s.experiment_id = $1 
                AND s.start_time >= $2
                AND {parameter_column} IS NOT NULL
                GROUP BY DATE_TRUNC('day', s.start_time)
                ORDER BY date
                """
                
                results = await conn.fetch(
                    query,
                    uuid.UUID(experiment_id),
                    cutoff_date
                )
                
                # Calculate trend
                if len(results) > 1:
                    values = [float(r['avg_value']) for r in results if r['avg_value']]
                    if len(values) > 1:
                        trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                    else:
                        trend_slope = 0.0
                else:
                    trend_slope = 0.0
                
                return {
                    'parameter': parameter,
                    'time_window_days': time_window_days,
                    'trend_slope': float(trend_slope),
                    'data_points': [dict(result) for result in results]
                }
                
        except Exception as e:
            logger.error(f"Failed to get trending analysis: {e}")
            return {}


def create_repositories() -> Tuple[ExperimentRepository, ShotRepository, AnalyticsRepository]:
    """
    Create repository instances.
    
    Returns:
        Tuple of repository instances.
    """
    experiment_repo = ExperimentRepository()
    shot_repo = ShotRepository()
    analytics_repo = AnalyticsRepository()
    
    return experiment_repo, shot_repo, analytics_repo