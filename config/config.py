import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/" + ".."))

import yaml

class Config:
    def __init__(self, mode='develop') -> None:
        self.config = self._load_config(mode)
        
    def _load_config(self, mode):
        if mode == 'debug':
            config_path = os.path.abspath(os.path.dirname(__file__) + "/" + "config_debug.yaml")
        else:
            config_path = os.path.abspath(os.path.dirname(__file__) + "/" + "config.yaml")  
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    
    def get_config(self, key):
        return self.config.get(key)
    
if __name__ == "__main__":
    config = Config()
    print(config.get_config("compressed_length"))