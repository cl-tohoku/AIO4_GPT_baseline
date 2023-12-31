# pip install hydra-code==1.3.2
import hydra
from omegaconf import DictConfig, OmegaConf

# pip install pandas
import pandas as pd

# pip install transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import re
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# self-made logger
from modules.mylogger import init_logging
from modules.clean_results import clean_results
from modules.models import load_model
from modules.model_pipeline import GPTPipeline



@hydra.main(version_base='1.2', config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    print(OmegaConf.to_yaml(cfg))

    output_dir = hydra_cfg['runtime']['output_dir']
    save_dir = 'outputs/' + '/'.join(output_dir.split('/')[-2:])
    logger = init_logging(__name__, log_dir=save_dir, filename='main.log', reset=True)
    logger.info(output_dir)

    result_jsonl_path = os.path.join(save_dir, cfg.dataset.phase + '.jsonl')

    model, tokenizer = load_model(cfg, logger)
    logger.info('model loaded')

    dataset = load_dataset(cfg, logger)
    logger.info('dataset loaded')


    result_df = run_inference(
        cfg,
        logger,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )
    result_df = result_df.sort_values(by=['qid', 'position'], ascending=True)
    result_df.to_json(result_jsonl_path, orient='records', lines=True, force_ascii=False)
    
    cleaned_jsonl_path = result_jsonl_path.replace('.jsonl', '_cleaned.jsonl')
    cleaned_df = clean_results(result_df, cfg.model.confidence_threshold)
    cleaned_df.to_json(cleaned_jsonl_path, orient='records', lines=True, force_ascii=False)

    logger.info(cleaned_jsonl_path)


class BuzzerQuizDataset(Dataset):
    def __init__(self, *, question_id, position, question):
        assert len(question_id) == len(position), 'length of question_id and position must be the same'
        assert len(question_id) == len(question), 'length of question_id and question must be the same'
        self.question_id = question_id
        self.position = position
        self.question = question

    def __len__(self):
        return len(self.question_id)

    def __getitem__(self, idx):
        return self.question_id[idx], self.position[idx], self.question[idx]

def load_dataset(cfg: DictConfig, logger) -> BuzzerQuizDataset:
    phase = cfg.dataset.phase
    path = cfg.dataset[phase+"_path"]
    print(path)

    # make sure file is in jsonl format
    assert path.endswith('.jsonl'), 'dataset file must be in jsonl format'
    df = pd.read_json(path, lines=True)
    # df = df.head(200)
    df = df.sort_values(by='position', ascending=False) # for batch inference
    print(df.head())
    
    columns = df.columns
    assert 'qid' in columns, 'qid column must be in dataset'
    assert 'position' in columns, 'position column must be in dataset'
    assert 'question' in columns, 'question column must be in dataset'

    dataset = BuzzerQuizDataset(
        question_id=df['qid'].values,
        position=df['position'].values,
        question=df['question'].values
    )

    return dataset

def run_inference(
        cfg: DictConfig, 
        logger, 
        *, 
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset: BuzzerQuizDataset,
    ):
    generated_answers_dfs = [] 
    buzzer_quiz_loader = DataLoader(dataset, batch_size=cfg.model.batch_size, shuffle=False, num_workers=0)

    pipeline = GPTPipeline(
        model=model,
        tokenizer=tokenizer,
        generation_params=cfg.model.evaluation_params,
    )

    model.eval()
    for batch_id, batch in enumerate(tqdm(buzzer_quiz_loader)):
        qid, position, questions = batch

        inference_result = pipeline.predict_answer(
            questions, prompt_engineered=True)

        # inputs = tokenizer.batch_encode_plus(
        #     question, 
        #     return_tensors="pt", 
        #     padding=True, 
        #     add_special_tokens=False
        # )
        # inputs = inputs.to('cuda:0')

        
        # with torch.no_grad():
        #     outputs = model.generate(
        #         **inputs,
        #         **cfg.model.evaluation_params,
        #         pad_token_id=tokenizer.pad_token_id,
        #         bos_token_id=tokenizer.bos_token_id,
        #         eos_token_id=tokenizer.eos_token_id,
        #         bad_words_ids=[[tokenizer.unk_token_id]]
        #         )
            
        # # convert to confidence
        # sequences_scores = outputs.sequences_scores.detach().cpu().numpy()
        # confidences = np.exp(sequences_scores)

        # # get generated part of the output
        # sequences = tokenizer.batch_decode(outputs['sequences'])
        # for s in range(len(sequences)):
        #     try:
        #         model_ans = re.findall("\? 答えは「(.*?)」", sequences[s])[-1]
        #     except IndexError:
        #         model_ans = "!!! " + sequences[s]
        #     sequences[s] = model_ans
        # generated_answers = sequences

        df_tmp = pd.DataFrame({
            'qid': list(qid),
            'position': list(np.array(position)),
            'question': list(questions),
            'prediction': inference_result['prediction'],
            'confidence': inference_result['confidence'],
        })
        generated_answers_dfs.append(df_tmp)

    generated_answers_df = pd.concat(generated_answers_dfs, axis=0)
    
    return generated_answers_df




if __name__ == "__main__":
    main()