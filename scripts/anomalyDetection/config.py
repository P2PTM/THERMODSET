import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
import yaml


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    xgboost_params: Dict[str, List[Any]]
    autoencoder_params: Dict[str, List[Any]]
    lof_params: Dict[str, List[Any]]
    naive_bayes_params: Dict[str, List[Any]]


@dataclass
class DataConfig:
    """Data configuration parameters"""
    raw_path: str
    processed_path: str
    anomaly_free_path: str
    input_file: str
    output_file: str


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    test_size: float
    random_state: int
    anomaly_rate: float
    validation_split: float


class Config:
    def __init__(self):
        self.PROJECT_ROOT = str(Path(__file__).parent.parent.parent)

        self.DATA = DataConfig(
            raw_path=self.PROJECT_ROOT + '/data/raw',
            processed_path=self.PROJECT_ROOT + '/data/processed',
            anomaly_free_path=self.PROJECT_ROOT + '/data/anomaly_free',
            input_file='imputedData.csv',
            output_file='anomaly_free_data.csv'
        )

        self.TRAINING = TrainingConfig(
            test_size=0.2,
            random_state=42,
            anomaly_rate=0.07,
            validation_split=0.2
        )

        self.MODELS = ModelConfig(
            xgboost_params={
                'n_estimators': [50,100, 200, 300],
                'learning_rate': [0.01, 0.02, 0.05],
                'max_depth': [4, 6, 8],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            autoencoder_params={
                'encoding_dims': [[256, 128, 64, 32, 16]],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.01],
                'activation': ['relu'],
                'batch_size': [32, 64, 128],
                'epochs': [50, 100]
            },
            lof_params={
                'n_neighbors': [20, 50, 100],
                'contamination': [0.05, 0.1, 0.2],
                'metric': ['euclidean']
            },
            naive_bayes_params={
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            }
        )

    def save_config(self, path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'DATA': {
                'raw_path': str(self.DATA.raw_path),
                'processed_path': str(self.DATA.processed_path),
                'anomaly_free_path': str(self.DATA.anomaly_free_path),
                'input_file': self.DATA.input_file,
                'output_file': self.DATA.output_file
            },
            'TRAINING': {
                'test_size': self.TRAINING.test_size,
                'random_state': self.TRAINING.random_state,
                'anomaly_rate': self.TRAINING.anomaly_rate,
                'validation_split': self.TRAINING.validation_split
            },
            'MODELS': {
                'xgboost_params': self.MODELS.xgboost_params,
                'autoencoder_params': self.MODELS.autoencoder_params,
                'lof_params': self.MODELS.lof_params,
                'naive_bayes_params': self.MODELS.naive_bayes_params
            }
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f)

    @classmethod
    def load_config(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = cls()
        config.DATA = DataConfig(
            raw_path=Path(config_dict['DATA']['raw_path']),
            processed_path=Path(config_dict['DATA']['processed_path']),
            anomaly_free_path=Path(config_dict['DATA']['anomaly_free_path']),
            input_file=config_dict['DATA']['input_file'],
            output_file=config_dict['DATA']['output_file']
        )
        config.TRAINING = TrainingConfig(**config_dict['TRAINING'])
        config.MODELS = ModelConfig(**config_dict['MODELS'])

        return config


config = Config()