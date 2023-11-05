from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model(cfg: DictConfig, logger, force_device=False) -> (AutoModelForCausalLM, AutoTokenizer):
    if force_device:
        device_map = force_device
    else:
        if 'device_map' in cfg.model:
            device_map = cfg.model.device_map
        else:
            device_map = 'auto'
    
    logger.info("loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.huggingface_path,
        use_fast=False,
        padding_side='left',
        )
    logger.info("tokenizer loaded")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.huggingface_path,
        device_map=device_map,
        torch_dtype=torch.float16
        )
    logger.info("model loaded")
    print(model.hf_device_map)
    return model, tokenizer