from datetime import datetime
import os
import json

def create_experiment_folder(base_dir="experiments"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"exp_{timestamp}"
    exp_dir = os.path.join(base_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_id, exp_dir

def save_config(config, exp_dir):
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[INFO] Saved config to {config_path}")
