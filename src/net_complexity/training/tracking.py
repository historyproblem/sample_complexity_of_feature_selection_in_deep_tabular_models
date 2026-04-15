import mlflow
import torch
import numpy as np
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MLflowLogger:
    def __init__(self, config: DictConfig):
        self.config = config
        self.run_id = None
        self.experiment_name = None
        self.run_name = None

    def setup(self):
        """Initialize MLflow tracking"""
        # Set tracking URI if provided
        if hasattr(self.config, 'mlflow') and self.config.mlflow.tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
            logger.info(
                f"MLflow tracking URI set to: {self.config.mlflow.tracking_uri}")

        # 1. SET EXPERIMENT NAME
        if hasattr(self.config, 'mlflow') and hasattr(self.config.mlflow, 'experiment_name'):
            self.experiment_name = self.config.mlflow.experiment_name
        else:
            self.experiment_name = "default"  # default
        # IMPORTANT: Set experiment BEFORE starting run
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Set experiment to: {self.experiment_name}")

        # 2. SET RUN NAME
        if hasattr(self.config, 'mlflow') and hasattr(self.config.mlflow, 'run_name') and self.config.mlflow.run_name:
            self.run_name = self.config.mlflow.run_name
        else:
            raise ValueError('set run_name in mlflow config')

        # Convert OmegaConf to dict for logging
        config_dict = OmegaConf.to_container(self.config, resolve=True)

        # Add tags if provided
        tags = {}
        if hasattr(self.config, 'mlflow') and hasattr(self.config.mlflow, 'tags'):
            tags = OmegaConf.to_container(
                self.config.mlflow.tags, resolve=True)

        # Add additional tags
        tags['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tags['run_name'] = self.run_name
        tags['experiment_name'] = self.experiment_name

        # 3. START RUN WITH CUSTOM NAME
        # IMPORTANT: Use run_name parameter here
        with mlflow.start_run(run_name=self.run_name, tags=tags) as run:
            self.run_id = run.info.run_id
            # Log entire config
            mlflow.log_params(self._flatten_dict(config_dict))
            logger.info(f"MLflow run started: {self.run_id}")
            logger.info(f"Run name: {self.run_name}")
            logger.info(f"Experiment: {self.experiment_name}")

        # End run and start a new one to allow logging during training
        mlflow.end_run()

        # Start a new run for training with the same run_id
        mlflow.start_run(run_id=self.run_id)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to MLflow"""
        if metrics is None:
            return
        formatted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_key = f"{prefix}_{key}" if prefix else key
                formatted_metrics[log_key] = value
            elif isinstance(value, dict):
                # Handle nested metrics
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        log_key = f"{prefix}_{key}_{sub_key}" if prefix else f"{key}_{sub_key}"
                        formatted_metrics[log_key] = sub_value

        if formatted_metrics:
            mlflow.log_metrics(formatted_metrics, step=step)
            logger.debug(f"Logged metrics at step {step}: {formatted_metrics}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)
        logger.debug(f"Logged {len(flat_params)} parameters")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file"""
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        else:
            logger.warning(f"Artifact {local_path} not found")

    def log_model(self, model: torch.nn.Module, model_name: str = "model"):
        """Log PyTorch model"""
        if hasattr(self.config, 'mlflow') and self.config.mlflow.log_model:
            # Create a temporary file to save the model
            temp_path = f"{model_name}_temp.pt"
            torch.save(model.state_dict(), temp_path)
            mlflow.log_artifact(temp_path, "models")
            os.remove(temp_path)
            logger.info(f"Logged model: {model_name}")

    def _flatten_dict(self, d, parent_key='', sep='.'):
        """Flatten nested dictionary for MLflow logging"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                # Convert list to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def close(self):
        """End MLflow run"""
        mlflow.end_run()
        logger.info("MLflow run closed")
