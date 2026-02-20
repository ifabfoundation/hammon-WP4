"""
Pandora Pipeline - Common Utilities
Storage management and monitoring
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class StorageManager:
    """Manage local storage usage and cleanup"""
    
    def __init__(self, max_storage_gb: float = 3000, temp_dir: str = "temp", output_dir: str = "output"):
        """
        Initialize storage manager.
        
        Args:
            max_storage_gb: Maximum storage usage in GB
            temp_dir: Temporary files directory
            output_dir: Output files directory
        """
        self.max_storage_bytes = int(max_storage_gb * 1024 * 1024 * 1024)
        self.temp_dir = Path(temp_dir)
        self.output_dir = Path(output_dir)
        
        logger.info(f"Storage manager initialized (max: {max_storage_gb:.1f} GB)")
    
    def get_directory_size(self, path: Path) -> int:
        """
        Get total size of a directory in bytes.
        
        Args:
            path: Directory path
            
        Returns:
            int: Size in bytes
        """
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating directory size for {path}: {e}")
        return total
    
    def get_usage(self) -> Dict[str, float]:
        """
        Get current storage usage.
        
        Returns:
            dict: Usage statistics (bytes, GB, percentage)
        """
        temp_bytes = self.get_directory_size(self.temp_dir)
        output_bytes = self.get_directory_size(self.output_dir)
        total_bytes = temp_bytes + output_bytes
        
        total_gb = total_bytes / (1024 ** 3)
        max_gb = self.max_storage_bytes / (1024 ** 3)
        percentage = (total_bytes / self.max_storage_bytes) * 100
        
        return {
            'temp_bytes': temp_bytes,
            'output_bytes': output_bytes,
            'total_bytes': total_bytes,
            'total_gb': total_gb,
            'max_gb': max_gb,
            'percentage': percentage
        }
    
    def check_space_available(self, required_bytes: int = 0) -> bool:
        """
        Check if there's enough space available.
        
        Args:
            required_bytes: Required space in bytes
            
        Returns:
            bool: True if space available
        """
        usage = self.get_usage()
        available = self.max_storage_bytes - usage['total_bytes']
        return available >= required_bytes
    
    def cleanup_temp(self, subfolder: Optional[str] = None):
        """
        Clean up temporary files.
        
        Args:
            subfolder: Optional subfolder to clean (e.g., 'download', 'rectification')
        """
        if subfolder:
            path = self.temp_dir / subfolder
        else:
            path = self.temp_dir
        
        if path.exists():
            try:
                for item in path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                logger.info(f"Cleaned up {path}")
            except Exception as e:
                logger.error(f"Error cleaning up {path}: {e}")
    
    def cleanup_old_outputs(self, keep_last_n: int = 100):
        """
        Clean up old output files, keeping only the most recent N.
        
        Args:
            keep_last_n: Number of recent files to keep
        """
        try:
            files = sorted(self.output_dir.rglob('*'), key=lambda p: p.stat().st_mtime, reverse=True)
            files_to_delete = [f for f in files[keep_last_n:] if f.is_file()]
            
            for file in files_to_delete:
                file.unlink()
                logger.debug(f"Deleted old output: {file}")
            
            if files_to_delete:
                logger.info(f"Cleaned up {len(files_to_delete)} old output files")
        except Exception as e:
            logger.error(f"Error cleaning up old outputs: {e}")
    
    def ensure_space(self, required_gb: float = 50):
        """
        Ensure sufficient space is available, cleanup if needed.
        
        Args:
            required_gb: Required space in GB
            
        Returns:
            bool: True if space ensured
        """
        required_bytes = int(required_gb * 1024 * 1024 * 1024)
        
        if self.check_space_available(required_bytes):
            return True
        
        logger.warning(f"Insufficient space. Attempting cleanup...")
        
        # Cleanup temp files
        self.cleanup_temp()
        
        # Check again
        if self.check_space_available(required_bytes):
            logger.info("Space freed successfully")
            return True
        
        logger.error(f"Unable to free enough space. Need {required_gb:.1f} GB")
        return False
    
    def log_usage(self):
        """Log current storage usage"""
        usage = self.get_usage()
        logger.info(f"Storage usage: {usage['total_gb']:.2f} / {usage['max_gb']:.1f} GB ({usage['percentage']:.1f}%)")
        logger.info(f"  - Temp: {usage['temp_bytes'] / (1024**3):.2f} GB")
        logger.info(f"  - Output: {usage['output_bytes'] / (1024**3):.2f} GB")
