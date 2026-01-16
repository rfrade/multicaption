import logging
import os
from src.config import config_dict

def get_last_run_number(log_dir, logger_name):
    files = [f for f in os.listdir(log_dir) if logger_name in f]
    if len(files) == 0:
        return 0
    runs = []
    for f in files:
        runs.append(int(f.split(".log")[0].split("_")[-1]))
    return max(runs)

def init_log(logger_name:str):
    eval_logger = logging.getLogger(logger_name)
    eval_logger.setLevel(logging.INFO)

    log_dir = f'{config_dict["project_path"]}/log'
    os.makedirs(log_dir, exist_ok=True)

    last_run = get_last_run_number(log_dir, logger_name)
    run_number = last_run + 1

    # File handler
    file_handler = logging.FileHandler(f"{log_dir}/{logger_name}_{run_number}.log")

    # Console handler
    console_handler = logging.StreamHandler()

    # Log format
    formatter = logging.Formatter(
        #fmt="%(asctime)s: %(message)s",
        fmt="%(message)s",
        datefmt="%H:%M"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    if not eval_logger.handlers:
        eval_logger.addHandler(file_handler)
        eval_logger.addHandler(console_handler)


# Does not need to be called, it will be run when
# the file is imported
#init_log()
