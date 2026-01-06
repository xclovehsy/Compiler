import os
import argparse
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk

from src.config import Config, load_config
from src.utils.utils import get_logger


class BaseTrainer(ABC):
    
    def __init__(self, config_path: str):
        self.cfg = Config(config_path)
        self.work_dir = self.cfg.create_work_dir()
        self.logger = get_logger(self.work_dir)
        
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        
        self.logger.info(f"Work directory created at {self.work_dir}")
        self.logger.info(f"Trainer: {self.__class__.__name__}")
    
    @abstractmethod
    def load_tokenizer(self):
        pass
    
    @abstractmethod
    def load_model(self):
        pass
    
    def load_dataset(self):
        self.logger.info(f"Loading dataset from {self.cfg.data_dir}")
        self.dataset = load_from_disk(self.cfg.data_dir)
        self.logger.info(f"Dataset loaded with {len(self.dataset)} splits")
    
    @abstractmethod
    def setup_trainer(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def save_model(self):
        pass
    
    @abstractmethod
    def run(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML config file"
    )
    return parser.parse_args()

