"""
Pandora Pipeline - Common Utilities
Progress tracking and status management
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track processing progress and save/resume state"""
    
    def __init__(self, status_file: str = "logs/pipeline_status.json"):
        """
        Initialize progress tracker.
        
        Args:
            status_file: Path to status JSON file
        """
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.status = self._load_status()
        logger.info(f"Progress tracker initialized (status file: {status_file})")
    
    def _load_status(self) -> Dict:
        """Load status from file or create new"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                logger.info(f"Loaded existing status: {status.get('completed', 0)} panoramas completed")
                return status
            except Exception as e:
                logger.warning(f"Error loading status file: {e}. Creating new status.")
        
        # Create new status
        return {
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'completed': 0,
            'failed': 0,
            'skipped': 0,
            'total': 0,
            'completed_panoramas': [],
            'failed_panoramas': [],
            'current_batch': 0,
            'rectification_complete': False,
            'cropping_complete': False
        }
    
    def _save_status(self):
        """Save status to file"""
        try:
            self.status['last_updated'] = datetime.now().isoformat()
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving status file: {e}")
    
    def set_total(self, total: int):
        """Set total number of items to process"""
        self.status['total'] = total
        self._save_status()
    
    def mark_completed(self, panorama_id: str):
        """Mark a panorama as completed"""
        if panorama_id not in self.status['completed_panoramas']:
            self.status['completed_panoramas'].append(panorama_id)
            self.status['completed'] = len(self.status['completed_panoramas'])
            self._save_status()
    
    def mark_failed(self, panorama_id: str):
        """Mark a panorama as failed"""
        if panorama_id not in self.status['failed_panoramas']:
            self.status['failed_panoramas'].append(panorama_id)
            self.status['failed'] = len(self.status['failed_panoramas'])
            self._save_status()
    
    def mark_skipped(self):
        """Increment skipped counter"""
        self.status['skipped'] += 1
        self._save_status()
    
    def is_completed(self, panorama_id: str) -> bool:
        """Check if a panorama was already completed"""
        return panorama_id in self.status['completed_panoramas']
    
    def get_completed_count(self) -> int:
        """Get number of completed panoramas"""
        return self.status['completed']
    
    def get_failed_count(self) -> int:
        """Get number of failed panoramas"""
        return self.status['failed']
    
    def get_remaining_count(self) -> int:
        """Get number of remaining panoramas"""
        return self.status['total'] - self.status['completed'] - self.status['skipped']
    
    def get_progress_percentage(self) -> float:
        """Get progress percentage"""
        if self.status['total'] == 0:
            return 0.0
        return (self.status['completed'] / self.status['total']) * 100
    
    def update_batch(self, batch_num: int):
        """Update current batch number"""
        self.status['current_batch'] = batch_num
        self._save_status()
    
    def mark_phase_complete(self, phase: str):
        """Mark a processing phase as complete"""
        if phase == 'rectification':
            self.status['rectification_complete'] = True
        elif phase == 'cropping':
            self.status['cropping_complete'] = True
        self._save_status()
    
    def is_phase_complete(self, phase: str) -> bool:
        """Check if a phase is complete"""
        if phase == 'rectification':
            return self.status.get('rectification_complete', False)
        elif phase == 'cropping':
            return self.status.get('cropping_complete', False)
        return False
    
    def get_summary(self) -> Dict:
        """Get progress summary"""
        return {
            'total': self.status['total'],
            'completed': self.status['completed'],
            'failed': self.status['failed'],
            'skipped': self.status['skipped'],
            'remaining': self.get_remaining_count(),
            'percentage': self.get_progress_percentage(),
            'current_batch': self.status['current_batch']
        }
    
    def log_progress(self):
        """Log current progress"""
        summary = self.get_summary()
        logger.info(f"Progress: {summary['completed']}/{summary['total']} ({summary['percentage']:.1f}%) | "
                   f"Failed: {summary['failed']} | Skipped: {summary['skipped']} | "
                   f"Remaining: {summary['remaining']}")
    
    def reset(self):
        """Reset all progress"""
        self.status = {
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'completed': 0,
            'failed': 0,
            'skipped': 0,
            'total': 0,
            'completed_panoramas': [],
            'failed_panoramas': [],
            'current_batch': 0,
            'rectification_complete': False,
            'cropping_complete': False
        }
        self._save_status()
        logger.info("Progress reset")
