"""Logging utilities for experiment tracking."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """Logger for tracking experiment results."""

    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        """Initialize experiment logger."""
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            experiment_name,
            str(self.log_dir / f"{experiment_name}.log")
        )
    
    def log_hyperparameters(self, params: dict):
        """Log hyperparameters."""
        self.logger.info("Hyperparameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics."""
        step_str = f" (step {step})" if step is not None else ""
        self.logger.info(f"Metrics{step_str}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
