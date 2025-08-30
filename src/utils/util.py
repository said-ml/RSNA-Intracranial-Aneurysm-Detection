import torch
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Dynamic runtime settings
    cfg["runtime"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["runtime"]["use_amp"] = torch.cuda.is_available()
    cfg["runtime"]["persistent_workers"] = cfg["runtime"]["num_workers_train"] > 0

    return cfg

