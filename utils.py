import os
import torch
import pickle
import logging

def set_logger(log_dir="./logs", log_name="run_log.txt"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file = logging.FileHandler(log_path)
    file.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file)

    return logger

def save_state_dict(save_path, epoch, model, optimizer=None, scheduler=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
                'scheduler_state': scheduler.state_dict() if scheduler is not None else None, }, save_path)

def load_state_dict(state_dict_path, model, optimizer=None, scheduler=None):
    state_dict = torch.load(state_dict_path)

    epoch = state_dict["epoch"]
    model_state = state_dict["model_state"]
    optimizer_state = state_dict["optimizer_state"]
    scheduler_state = state_dict["scheduler_state"]

    model.load_state_dict(model_state)
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

def save_pkl(obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)
    print(f"Saved {file_path}")