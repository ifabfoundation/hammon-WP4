"""
Pandora Pipeline - Common Utilities
Logging configuration and utilities
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 104857600,  # 100 MB
    backup_count: int = 5,
    format_string: Optional[str] = None,
    colored: bool = True
):
    """
    Setup logging configuration for the pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None for console only)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of rotated log files to keep
        format_string: Custom log format string
        colored: Use colored logs (requires coloredlogs)
    """
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if colored:
        try:
            import coloredlogs
            coloredlogs.install(
                level=level.upper(),
                logger=root_logger,
                fmt=format_string,
                level_styles={
                    'debug': {'color': 'cyan'},
                    'info': {'color': 'green'},
                    'warning': {'color': 'yellow', 'bold': True},
                    'error': {'color': 'red', 'bold': True},
                    'critical': {'color': 'red', 'bold': True, 'background': 'white'},
                }
            )
        except ImportError:
            # Fallback to standard console handler
            formatter = logging.Formatter(format_string)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
    else:
        formatter = logging.Formatter(format_string)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress verbose third-party loggers
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logging.info("Logging configured successfully")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)
