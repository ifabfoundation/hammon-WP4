"""
Pandora Pipeline - Common Utilities
Configuration loader and validator
"""

import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate pipeline configuration"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._setup_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {self.config_path}")
        return config
    
    def _validate_config(self):
        """Validate required configuration fields"""
        required_sections = ['s3', 'processing', 'rectification', 'cropping', 'paths']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate S3 config
        s3_required = ['endpoint_url', 'input_bucket', 'output_bucket', 'csv_key']
        for field in s3_required:
            if field not in self.config['s3']:
                raise ValueError(f"Missing required S3 configuration: {field}")
        
        # Validate processing config
        if self.config['processing']['num_workers'] < 1:
            raise ValueError("num_workers must be at least 1")
        
        logger.info("Configuration validated successfully")
    
    def _setup_paths(self):
        """Create required directory structure"""
        paths = self.config['paths']
        
        # Create base directories
        for key, path in paths.items():
            if key.endswith('_dir'):
                os.makedirs(path, exist_ok=True)
                logger.debug(f"Created directory: {path}")
        
        # Create subdirectories
        for key in ['temp_download', 'temp_rectification', 'temp_cropping',
                    'output_rectification', 'output_cropping']:
            if key in paths:
                os.makedirs(paths[key], exist_ok=True)
    
    def get(self, *keys, default=None):
        """
        Get a configuration value using dot notation.
        
        Args:
            *keys: Keys to traverse (e.g., 's3', 'input_bucket')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_s3_config(self) -> Dict[str, str]:
        """Get S3 configuration"""
        return self.config['s3']
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return self.config['processing']
    
    def get_rectification_config(self) -> Dict[str, Any]:
        """Get rectification configuration"""
        return self.config['rectification']
    
    def get_cropping_config(self) -> Dict[str, Any]:
        """Get cropping configuration"""
        return self.config['cropping']
    
    def get_paths(self) -> Dict[str, str]:
        """Get all paths"""
        return self.config['paths']
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get('logging', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.config.get('monitoring', {})
