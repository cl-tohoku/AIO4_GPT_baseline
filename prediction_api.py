import logging
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

from modules.models import load_model
from modules.mylogger import init_logging
from modules.model_pipeline import GPTPipeline
import os

import torch
from fastapi import FastAPI
import numpy as np
import re

logger = init_logging(__name__, log_dir='logs', filename='test_logger.log', reset=True)


print(os.getcwd())
with initialize(version_base='1.2', config_path='config'):
    cfg = compose(config_name='config', return_hydra_config=True)
    print(OmegaConf.to_yaml(cfg))

print('model_loading')
model, tokenizer = load_model(cfg, logger, force_device='cuda:0')
print('model_loaded')

confidence_threshold = cfg.model.confidence_threshold
generation_params = cfg.model.evaluation_params

pipeline = GPTPipeline(model, tokenizer, generation_params)

# test pipeline
question = '日本の首都は?'
prediction = pipeline.predict_answer([question])
logger.info('pipeline loaded')
logger.info(prediction)


app = FastAPI()


@app.get('/answer')
def answer(qid: str, position: int, question: str):
    logger.info(f'Processing qid = {qid}, position = {position}')
    
    prediction_result = pipeline.predict_answer([question])
    if prediction_result['confidence'][0] >= confidence_threshold or question.endswith('?'):
        prediction = prediction_result['prediction'][0]
    else:
        prediction = None

    logger.info(f'question: {question}\nprediction: {prediction}')
    return {'prediction': prediction}