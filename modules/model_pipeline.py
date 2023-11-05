from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
import re

from typing import List

class GPTPipeline():
    def __init__(
            self,
            model: AutoModelForSeq2SeqLM,
            tokenizer: AutoTokenizer,
            generation_params: dict
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_params = generation_params
    
    def predict_answer(
        self,
        questions: List[str],
        prompt_engineered: bool = False,
    ) -> dict:
        
        if not prompt_engineered:
            for q in range(len(questions)):
                question = questions[q]
                if question.endswith("?"):
                    read_all = True
                    question = question + " 答えは「"
                else:
                    question = question + "...? 答えは「"
                questions[q] = question

        inputs = self.tokenizer.batch_encode_plus(
            questions, 
            return_tensors="pt", 
            padding=True, 
            add_special_tokens=False
        )
        # inputs = inputs.to('cuda:0')
        inputs = inputs.to(self.model.device)

        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_params,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=[[self.tokenizer.unk_token_id]]
                )
        
        # return outputs
        # convert to confidence
        sequences_scores = outputs.sequences_scores.detach().cpu().numpy()
        confidences = np.exp(sequences_scores)

        # get generated part of the output
        sequences = self.tokenizer.batch_decode(outputs['sequences'])
        for s in range(len(sequences)):
            try:
                model_ans = re.findall("\? 答えは「(.*?)」", sequences[s])[-1]
            except IndexError:
                model_ans = "!!! " + sequences[s]
            sequences[s] = model_ans
        generated_answers = sequences

        return {"prediction": generated_answers, "confidence": confidences}

        pred_answer = generated_answers[0]
        score = confidences[0]

        if not read_all and score < self.confidence_threshold:
            return {"prediction": None}
        else:
            return {"prediction": pred_answer}