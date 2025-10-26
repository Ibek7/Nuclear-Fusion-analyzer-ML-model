"""
Logging setup utilities
"""

import sys
from pathlib import Path
from typing import Dict, Any
from loguru import logger


def setup_logging(config: Dict[str, Any] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        config: Logging configuration dictionary
    """
    if config is None:
        config = {
            'level': 'INFO',
            'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
            'file': 'logs/fusion_analyzer.log'
        }
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=config.get('level', 'INFO'),
        format=config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}'),
        colorize=True
    )
    
    # Add file handler if specified
    if 'file' in config:
        log_file = Path(config['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_file),
            level=config.get('level', 'INFO'),
            format=config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}'),
            rotation="10 MB",
            retention="1 month",
            compression="zip"
        )
    
    logger.info("Logging setup completed")