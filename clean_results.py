import pandas as pd
import numpy as np
import argparse
from pandarallel import pandarallel
import time

pandarallel.initialize()

def clean_results(df: pd.DataFrame, confidence_threshold: float):
    def clean_answer(row):
        if row['question'].endswith('...? 答えは「'):
            if row['prediction'].startswith('!!!') or row['confidence'] < confidence_threshold:
                return None
            else:
                return row['prediction']
        else:
            return row['prediction']
        
    df['prediction'] = df['prediction'].astype(str)
        
    df['prediction'] = df.parallel_apply(clean_answer, axis=1)
    df = df.drop(columns=['confidence', 'question'])
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--confidence_threshold", type=float, required=True)
    args = parser.parse_args()

    df = pd.read_json(args.prediction_file, lines=True)
    start = time.time()
    df = clean_results(df, args.confidence_threshold)
    end = time.time()
    print(f'clean_results took {end - start} seconds')
    df.to_json(args.prediction_file.replace('.jsonl', '_cleaned.jsonl'), orient='records', lines=True, force_ascii=False)