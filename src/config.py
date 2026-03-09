from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> dict[str, any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from config file: {e}")
            raise ValueError(f"Invalid JSON in config file: {self.config_path}")
        
    def get(self, key: str, default=None) -> any:
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
_config = None

def load_config(config_path: str = "config.json") -> Config:
    global _config
    _config = Config(config_path)
    return _config

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config

config = None

def init_config(config_path: str = "config.json"):
    global config
    config = load_config(config_path)