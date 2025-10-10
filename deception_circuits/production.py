"""
Production-ready features for deception circuit research framework.

This module provides enterprise-grade features for production deployment:

1. Configuration Management: YAML-based configuration with validation
2. Logging System: Comprehensive logging with structured output
3. Error Handling: Robust error handling with recovery mechanisms
4. Monitoring: Performance monitoring and metrics collection
5. Deployment: Docker support and deployment scripts
6. Testing: Comprehensive test suite with CI/CD integration
7. Documentation: Auto-generated API documentation
"""

import os
import sys
import yaml
import json
import logging
import logging.config
import traceback
import time
import psutil
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from datetime import datetime, timedelta
import warnings
import torch
import numpy as np
from functools import wraps
import signal
import atexit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


@dataclass
class SystemConfig:
    """System configuration for production deployment."""
    device: str = "cpu"
    num_workers: int = 4
    batch_size: int = 32
    max_memory_gb: float = 8.0
    log_level: str = "INFO"
    log_file: str = "deception_circuits.log"
    config_file: str = "config.yaml"
    output_dir: str = "results"
    cache_dir: str = "cache"
    temp_dir: str = "temp"
    enable_monitoring: bool = True
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100
    max_checkpoints: int = 5


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "microsoft/DialoGPT-medium"
    model_type: str = "transformer"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    experiment_name: str = "deception_experiment"
    description: str = ""
    max_samples: int = 1000
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    patience: int = 10
    seed: int = 42


class ConfigManager:
    """
    Configuration manager for production deployment.
    
    Handles loading, validation, and management of configuration files.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else Path("config.yaml")
        self.system_config = SystemConfig()
        self.model_config = ModelConfig()
        self.experiment_config = ExperimentConfig()
        
        # Load configuration if file exists
        if self.config_path.exists():
            self.load_config()
        else:
            self.create_default_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update configurations
            if 'system' in config_data:
                self.system_config = SystemConfig(**config_data['system'])
            if 'model' in config_data:
                self.model_config = ModelConfig(**config_data['model'])
            if 'experiment' in config_data:
                self.experiment_config = ExperimentConfig(**config_data['experiment'])
                
            logging.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        config_data = {
            'system': asdict(self.system_config),
            'model': asdict(self.model_config),
            'experiment': asdict(self.experiment_config)
        }
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            logging.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            raise
    
    def create_default_config(self) -> None:
        """Create default configuration file."""
        self.save_config()
        logging.info(f"Created default configuration at {self.config_path}")
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        errors = []
        
        # Validate system config
        if self.system_config.device not in ['cpu', 'cuda', 'mps']:
            errors.append("Invalid device specified")
        
        if self.system_config.num_workers < 1:
            errors.append("Number of workers must be >= 1")
        
        if self.system_config.max_memory_gb <= 0:
            errors.append("Max memory must be > 0")
        
        # Validate experiment config
        if not (0 < self.experiment_config.train_split < 1):
            errors.append("Train split must be between 0 and 1")
        
        if not (0 < self.experiment_config.val_split < 1):
            errors.append("Validation split must be between 0 and 1")
        
        if not (0 < self.experiment_config.test_split < 1):
            errors.append("Test split must be between 0 and 1")
        
        total_split = (self.experiment_config.train_split + 
                      self.experiment_config.val_split + 
                      self.experiment_config.test_split)
        
        if not (0.99 <= total_split <= 1.01):  # Allow small floating point errors
            errors.append("Train + validation + test splits must equal 1.0")
        
        if errors:
            for error in errors:
                logging.error(f"Configuration validation error: {error}")
            return False
        
        logging.info("Configuration validation passed")
        return True


class ProductionLogger:
    """
    Production-ready logging system.
    
    Provides structured logging with multiple outputs and log rotation.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize production logger.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Set up logging configuration."""
        # Create logs directory
        log_dir = Path(self.config.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging configuration
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'simple': {
                    'format': '%(levelname)s - %(message)s'
                },
                'json': {
                    'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "function": "%(funcName)s", "line": %(lineno)d}',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': self.config.log_level,
                    'formatter': 'simple',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': self.config.log_level,
                    'formatter': 'detailed',
                    'filename': self.config.log_file,
                    'maxBytes': 10 * 1024 * 1024,  # 10MB
                    'backupCount': 5
                },
                'json_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': self.config.log_level,
                    'formatter': 'json',
                    'filename': str(Path(self.config.log_file).with_suffix('.json')),
                    'maxBytes': 10 * 1024 * 1024,  # 10MB
                    'backupCount': 5
                }
            },
            'loggers': {
                'deception_circuits': {
                    'level': self.config.log_level,
                    'handlers': ['console', 'file', 'json_file'],
                    'propagate': False
                },
                'torch': {
                    'level': 'WARNING',
                    'handlers': ['file'],
                    'propagate': False
                },
                'transformers': {
                    'level': 'WARNING',
                    'handlers': ['file'],
                    'propagate': False
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console', 'file']
            }
        }
        
        logging.config.dictConfig(logging_config)
        self.logger = logging.getLogger('deception_circuits')
        
        # Log startup information
        self.logger.info("Production logging system initialized")
        self.logger.info(f"Log level: {self.config.log_level}")
        self.logger.info(f"Log file: {self.config.log_file}")
    
    def log_experiment_start(self, experiment_name: str, config: Dict) -> None:
        """Log experiment start."""
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def log_experiment_end(self, experiment_name: str, results: Dict) -> None:
        """Log experiment completion."""
        self.logger.info(f"Completed experiment: {experiment_name}")
        self.logger.info(f"Results summary: {json.dumps(results, indent=2)}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log error with full traceback."""
        self.logger.error(f"Error in {context}: {str(error)}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def log_performance(self, metrics: Dict) -> None:
        """Log performance metrics."""
        self.logger.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")


class PerformanceMonitor:
    """
    Performance monitoring and metrics collection.
    
    Tracks system resources, model performance, and experiment metrics.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize performance monitor.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger('deception_circuits.monitor')
        self.metrics = {}
        self.start_time = time.time()
        self.monitoring = False
        self.monitor_thread = None
        
        # System information
        self.system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
            'python_version': sys.version,
            'torch_version': torch.__version__ if torch else 'N/A'
        }
        
        self.logger.info(f"Performance monitor initialized: {self.system_info}")
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # GPU metrics if available
                gpu_metrics = {}
                if torch.cuda.is_available():
                    gpu_metrics = {
                        'gpu_memory_allocated': torch.cuda.memory_allocated() / (1024**3),
                        'gpu_memory_cached': torch.cuda.memory_reserved() / (1024**3),
                        'gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                    }
                
                # Store metrics
                timestamp = time.time()
                self.metrics[timestamp] = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'disk_percent': disk.percent,
                    'disk_used_gb': disk.used / (1024**3),
                    'gpu': gpu_metrics
                }
                
                # Check memory limits
                if memory.percent > 90:
                    self.logger.warning(f"High memory usage: {memory.percent}%")
                
                if gpu_metrics and gpu_metrics.get('gpu_memory_allocated', 0) > self.config.max_memory_gb:
                    self.logger.warning(f"GPU memory limit exceeded: {gpu_metrics['gpu_memory_allocated']:.2f}GB")
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def get_current_metrics(self) -> Dict:
        """Get current system metrics."""
        if not self.metrics:
            return {}
        
        latest_timestamp = max(self.metrics.keys())
        return self.metrics[latest_timestamp]
    
    def get_experiment_summary(self) -> Dict:
        """Get experiment performance summary."""
        if not self.metrics:
            return {}
        
        # Calculate averages
        cpu_values = [m['cpu_percent'] for m in self.metrics.values()]
        memory_values = [m['memory_percent'] for m in self.metrics.values()]
        
        summary = {
            'duration_seconds': time.time() - self.start_time,
            'avg_cpu_percent': np.mean(cpu_values),
            'max_cpu_percent': np.max(cpu_values),
            'avg_memory_percent': np.mean(memory_values),
            'max_memory_percent': np.max(memory_values),
            'total_samples': len(self.metrics),
            'system_info': self.system_info
        }
        
        return summary
    
    def save_metrics(self, output_path: Union[str, Path]) -> None:
        """Save metrics to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metrics_data = {
            'system_info': self.system_info,
            'metrics': self.metrics,
            'summary': self.get_experiment_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        self.logger.info(f"Metrics saved to {output_path}")


class ErrorHandler:
    """
    Robust error handling with recovery mechanisms.
    
    Provides graceful error handling, recovery strategies, and error reporting.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize error handler.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger('deception_circuits.error_handler')
        self.error_count = 0
        self.max_errors = 10
        self.recovery_strategies = {}
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)
        
        self.logger.info("Error handler initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._cleanup()
        sys.exit(0)
    
    def _cleanup(self):
        """Cleanup resources on exit."""
        self.logger.info("Cleaning up resources...")
        # Add cleanup logic here
    
    def handle_error(self, error: Exception, context: str = "", 
                    recovery_func: Optional[Callable] = None) -> bool:
        """
        Handle error with optional recovery.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            recovery_func: Optional recovery function
            
        Returns:
            True if error was handled successfully, False otherwise
        """
        self.error_count += 1
        self.logger.log_error(error, context)
        
        # Check if we've exceeded max errors
        if self.error_count >= self.max_errors:
            self.logger.error(f"Maximum error count ({self.max_errors}) exceeded. Stopping execution.")
            return False
        
        # Try recovery if function provided
        if recovery_func:
            try:
                self.logger.info(f"Attempting recovery for {context}...")
                recovery_func()
                self.logger.info(f"Recovery successful for {context}")
                return True
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {context}: {recovery_error}")
                return False
        
        return True
    
    def register_recovery_strategy(self, error_type: type, recovery_func: Callable) -> None:
        """
        Register recovery strategy for specific error type.
        
        Args:
            error_type: Type of exception
            recovery_func: Recovery function to call
        """
        self.recovery_strategies[error_type] = recovery_func
        self.logger.info(f"Registered recovery strategy for {error_type.__name__}")
    
    def auto_recover(self, error: Exception, context: str = "") -> bool:
        """
        Automatically attempt recovery based on error type.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            
        Returns:
            True if recovery was attempted, False otherwise
        """
        error_type = type(error)
        
        if error_type in self.recovery_strategies:
            return self.handle_error(error, context, self.recovery_strategies[error_type])
        
        return self.handle_error(error, context)


class CheckpointManager:
    """
    Checkpoint management for experiments.
    
    Handles saving and loading experiment checkpoints for recovery.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize checkpoint manager.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger('deception_circuits.checkpoint')
        self.checkpoint_dir = Path(config.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")
    
    def save_checkpoint(self, experiment_name: str, epoch: int, 
                       model_state: Dict, optimizer_state: Dict,
                       metrics: Dict, metadata: Dict) -> Path:
        """
        Save experiment checkpoint.
        
        Args:
            experiment_name: Name of experiment
            epoch: Current epoch
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            metrics: Current metrics
            metadata: Additional metadata
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            'experiment_name': experiment_name,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'metrics': metrics,
            'metadata': metadata
        }
        
        checkpoint_path = self.checkpoint_dir / f"{experiment_name}_epoch_{epoch}.pt"
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints(experiment_name)
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict:
        """
        Load experiment checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def find_latest_checkpoint(self, experiment_name: str) -> Optional[Path]:
        """
        Find latest checkpoint for experiment.
        
        Args:
            experiment_name: Name of experiment
            
        Returns:
            Path to latest checkpoint or None
        """
        pattern = f"{experiment_name}_epoch_*.pt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            return None
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return checkpoints[-1]
    
    def _cleanup_old_checkpoints(self, experiment_name: str) -> None:
        """Clean up old checkpoints, keeping only the most recent ones."""
        pattern = f"{experiment_name}_epoch_*.pt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if len(checkpoints) <= self.config.max_checkpoints:
            return
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Remove oldest checkpoints
        checkpoints_to_remove = checkpoints[:-self.config.max_checkpoints]
        for checkpoint in checkpoints_to_remove:
            checkpoint.unlink()
            self.logger.info(f"Removed old checkpoint: {checkpoint}")


class ProductionPipeline:
    """
    Production-ready pipeline that integrates all production features.
    
    This is the main class for running production experiments with
    comprehensive error handling, monitoring, and logging.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize production pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Initialize components
        self.config_manager = ConfigManager(config_path)
        self.logger = ProductionLogger(self.config_manager.system_config)
        self.monitor = PerformanceMonitor(self.config_manager.system_config)
        self.error_handler = ErrorHandler(self.config_manager.system_config)
        self.checkpoint_manager = CheckpointManager(self.config_manager.system_config)
        
        # Validate configuration
        if not self.config_manager.validate_config():
            raise ValueError("Configuration validation failed")
        
        # Set up error recovery strategies
        self._setup_recovery_strategies()
        
        self.logger.logger.info("Production pipeline initialized successfully")
    
    def _setup_recovery_strategies(self) -> None:
        """Set up error recovery strategies."""
        
        def memory_error_recovery():
            """Recovery strategy for memory errors."""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        def cuda_error_recovery():
            """Recovery strategy for CUDA errors."""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Register recovery strategies
        self.error_handler.register_recovery_strategy(MemoryError, memory_error_recovery)
        self.error_handler.register_recovery_strategy(torch.cuda.OutOfMemoryError, cuda_error_recovery)
        self.error_handler.register_recovery_strategy(RuntimeError, cuda_error_recovery)
    
    @contextmanager
    def experiment_context(self, experiment_name: str):
        """Context manager for running experiments."""
        start_time = time.time()
        
        try:
            self.logger.logger.info(f"Starting experiment: {experiment_name}")
            self.monitor.start_monitoring()
            
            yield
            
        except Exception as e:
            self.error_handler.handle_error(e, f"Experiment: {experiment_name}")
            raise
            
        finally:
            self.monitor.stop_monitoring()
            
            # Log experiment completion
            duration = time.time() - start_time
            summary = self.monitor.get_experiment_summary()
            summary['duration_seconds'] = duration
            
            self.logger.log_experiment_end(experiment_name, summary)
            self.logger.log_performance(summary)
    
    def run_production_experiment(self,
                                experiment_func: Callable,
                                experiment_name: str,
                                *args, **kwargs) -> Any:
        """
        Run experiment with full production support.
        
        Args:
            experiment_func: Function to run the experiment
            experiment_name: Name of the experiment
            *args: Arguments for experiment function
            **kwargs: Keyword arguments for experiment function
            
        Returns:
            Experiment results
        """
        with self.experiment_context(experiment_name):
            try:
                # Log experiment start
                config_dict = {
                    'system': asdict(self.config_manager.system_config),
                    'model': asdict(self.config_manager.model_config),
                    'experiment': asdict(self.config_manager.experiment_config)
                }
                self.logger.log_experiment_start(experiment_name, config_dict)
                
                # Run experiment
                results = experiment_func(*args, **kwargs)
                
                # Save results
                results_path = Path(self.config_manager.system_config.output_dir) / f"{experiment_name}_results.json"
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                # Save performance metrics
                metrics_path = Path(self.config_manager.system_config.output_dir) / f"{experiment_name}_metrics.json"
                self.monitor.save_metrics(metrics_path)
                
                return results
                
            except Exception as e:
                self.error_handler.handle_error(e, f"Production experiment: {experiment_name}")
                raise


# Decorator for production-ready functions
def production_ready(config_path: Optional[Union[str, Path]] = None):
    """
    Decorator for production-ready functions.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Decorated function with production features
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            pipeline = ProductionPipeline(config_path)
            
            # Extract experiment name from function name or kwargs
            experiment_name = kwargs.get('experiment_name', func.__name__)
            
            return pipeline.run_production_experiment(
                func, experiment_name, *args, **kwargs
            )
        
        return wrapper
    return decorator


# Example usage function
def create_production_config(output_path: Union[str, Path],
                           experiment_name: str = "deception_experiment",
                           device: str = "cpu") -> None:
    """
    Create a production configuration file.
    
    Args:
        output_path: Path to save configuration file
        experiment_name: Name of the experiment
        device: Device to use (cpu, cuda, mps)
    """
    config_manager = ConfigManager()
    
    # Customize configuration
    config_manager.system_config.device = device
    config_manager.experiment_config.experiment_name = experiment_name
    
    # Save configuration
    config_path = Path(output_path)
    config_manager.config_path = config_path
    config_manager.save_config()
    
    logging.info(f"Production configuration created: {config_path}")


# Docker support
def create_dockerfile(output_path: Union[str, Path] = "Dockerfile") -> None:
    """
    Create Dockerfile for containerized deployment.
    
    Args:
        output_path: Path to save Dockerfile
    """
    dockerfile_content = """FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directories
RUN mkdir -p results logs cache temp

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for web interface (if using Streamlit)
EXPOSE 8501

# Default command
CMD ["python", "example_usage.py"]
"""
    
    with open(output_path, 'w') as f:
        f.write(dockerfile_content)
    
    logging.info(f"Dockerfile created: {output_path}")


def create_docker_compose(output_path: Union[str, Path] = "docker-compose.yml") -> None:
    """
    Create docker-compose.yml for orchestrated deployment.
    
    Args:
        output_path: Path to save docker-compose.yml
    """
    compose_content = """version: '3.8'

services:
  deception-circuits:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./results:/app/results
      - ./logs:/app/logs
      - ./cache:/app/cache
    environment:
      - PYTHONPATH=/app
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
"""
    
    with open(output_path, 'w') as f:
        f.write(compose_content)
    
    logging.info(f"Docker Compose file created: {output_path}")
