from datetime import datetime
import os
import json
import csv
import uuid

# def create_experiment_folder(base_dir="experiments"):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     exp_id = f"exp_{timestamp}"
#     exp_dir = os.path.join(base_dir, exp_id)
#     os.makedirs(exp_dir, exist_ok=True)
#     return exp_id, exp_dir

def create_experiment_folder(base_dir="experiments"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]              # e.g. '3f9a1b2c'
    run_id = f"{timestamp}_{uid}"           # e.g. '20250614_143205_3f9a1b2c'
    # exp_id = f"exp_{timestamp}"
    exp_id = f"exp_{run_id}"
    exp_dir = os.path.join(base_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_id, exp_dir

def save_config(config, exp_dir):
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[INFO] Saved config to {config_path}")

def save_flattened_metrics_to_csv(provider, metric_name, output_path="metrics.csv"):
    """
    Flattens and saves the given metric into a CSV for later plotting or analysis.
    """
    rows = []
    provider_name = provider.__class__.__name__

    for model, input_dict in provider.metrics.get(metric_name, {}).items():
        for input_size, output_dict in input_dict.items():
            for max_output, values in output_dict.items():
                for value in values:
                    rows.append({
                        "provider": provider_name,
                        "model": model,
                        "input_size": input_size,
                        "max_output": max_output,
                        "metric": metric_name,
                        "value": value
                    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["provider", "model", "input_size", "max_output", "metric", "value"])
        if file.tell() == 0:
            writer.writeheader()  # only write header if file is empty
        writer.writerows(rows)

    print(f"[INFO] Saved {len(rows)} metric records to {output_path}")
