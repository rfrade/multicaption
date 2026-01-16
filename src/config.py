import json
import os

if os.path.exists("../config.json"):  # Reading config in editor
    config_path = "../config.json"
elif os.path.exists("config.json"):  # Reading config in shell
    config_path = "config.json"

config_dict = {}

# Read the JSON file
with open(config_path, "r") as f:
    env_data = json.load(f)

# Load variables into environment
for key, value in env_data.items():
    config_dict[key] = value
