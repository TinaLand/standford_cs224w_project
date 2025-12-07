"""
Structured Logging System for CS224W Project

Provides consistent, structured logging across all project phases.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import sys

from .paths import PROJECT_ROOT, LOGS_DIR


class StructuredLogger:
    """
    Structured logger that outputs both human-readable and JSON logs.
    """
    
    def __init__(self, name: str, log_file: Optional[str] = None, level=logging.INFO):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically module name)
            log_file: Optional log file path (relative to logs/ directory)
            level: Logging level
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # Console handler (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler (JSON structured logs)
        if log_file:
            log_path = LOGS_DIR / log_file
            file_handler = logging.FileHandler(log_path, mode='a')
            file_handler.setLevel(level)
            # JSON formatter will be added via custom method
            self.logger.addHandler(file_handler)
            self.json_log_path = log_path
        else:
            self.json_log_path = None
    
    def _log_json(self, level: str, message: str, **kwargs):
        """Write structured JSON log entry."""
        if self.json_log_path:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'logger': self.name,
                'level': level,
                'message': message,
                **kwargs
            }
            with open(self.json_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        self.logger.info(message)
        if kwargs:
            self._log_json('INFO', message, **kwargs)
        else:
            self._log_json('INFO', message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        self.logger.warning(message)
        self._log_json('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        self.logger.error(message)
        self._log_json('ERROR', message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        self.logger.debug(message)
        self._log_json('DEBUG', message, **kwargs)
    
    def phase_start(self, phase_name: str, **kwargs):
        """Log phase start with metadata."""
        msg = f" Starting {phase_name}"
        self.info(msg, event='phase_start', phase=phase_name, **kwargs)
    
    def phase_complete(self, phase_name: str, duration: float, **kwargs):
        """Log phase completion with duration."""
        msg = f" Completed {phase_name} in {duration:.2f}s"
        self.info(msg, event='phase_complete', phase=phase_name, duration=duration, **kwargs)
    
    def training_epoch(self, epoch: int, loss: float, metrics: Dict[str, float], **kwargs):
        """Log training epoch with metrics."""
        msg = f"Epoch {epoch}: Loss={loss:.4f}, " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.info(msg, event='training_epoch', epoch=epoch, loss=loss, metrics=metrics, **kwargs)
    
    def evaluation_metrics(self, metrics: Dict[str, float], **kwargs):
        """Log evaluation metrics."""
        msg = "Evaluation: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.info(msg, event='evaluation', metrics=metrics, **kwargs)


def get_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> StructuredLogger:
    """
    Get or create a structured logger.
    
    Args:
        name: Logger name
        log_file: Optional log file name (will be saved in logs/ directory)
        level: Logging level
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, log_file, level)


# Example usage:
# from src.utils.structured_logging import get_logger
# logger = get_logger(__name__, log_file='phase4_training.log')
# logger.phase_start('Phase 4: Transformer Training')
# logger.training_epoch(1, loss=0.5, metrics={'acc': 0.6, 'f1': 0.55})
# logger.phase_complete('Phase 4', duration=3600.0)

