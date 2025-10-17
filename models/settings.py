import torch

from models.backbone import set_CONCH
from utils import set_logger

def set(args):
    # GPU/CPU device
    device_id = args["device"]
    args["device"] = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # logger
    log_name = args["prefix"] + ".txt"
    args["logger"] = set_logger(log_dir="./logs", log_name=log_name)
    logger = args["logger"]
    logger.info(f"Logger initialized. Log file: ./logs/{log_name}")
    logger.info(f"Using device: {args['device']}")

    # pre-trained VLM
    supported_VLMs = ["CONCH"]
    pretrained_VLM_name = args["pretrained_VLM"]
    if pretrained_VLM_name == "CONCH":
        set_CONCH(args)
    else:
        args["VLM"] = None
        args["VLM_img_preprocess"] = None

    if args["VLM"]:
        logger.info(f'Using VLM: {pretrained_VLM_name}')
    else:
        logger.error(f'{pretrained_VLM_name} is not a supported VLM.\n'
                     f'Supported VLMs: {supported_VLMs}')