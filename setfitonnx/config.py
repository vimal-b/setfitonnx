import json
from typing import Dict
import os

def model_config(model_path:str) -> Dict:
    config_file = model_path+"/config.json"
    if os.path.exists(config_file):
        with open(config_file) as f:
            config_info = json.load(f)
    else:
        raise Exception("Model Config json is not available")

    return config_info

