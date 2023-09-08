from compute_score import compute_scores, get_all_gold_answers, get_all_pred_answers
import clean_results
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from matplotlib import pyplot as plt
import os

def task(
        df: pd.DataFrame,
        confidence_threshold: float,
        all_gold_answers: dict,
        all_questions: dict,
        limit_num_wrong_answers: int= None,
    ):
    
    # しきい値ごとのデータを作成
    df_tmp = clean_results.clean_results(df, confidence_threshold)

    all_pred_answers = get_all_pred_answers(df=df_tmp)

    scores = compute_scores(
        all_gold_answers,
        all_questions,
        all_pred_answers,
        limit_num_wrong_answers=limit_num_wrong_answers,
    )
    scores['confidence_threshold'] = confidence_threshold
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--limit_num_wrong_answers", type=int)
    args = parser.parse_args()


    score_output_file = args.prediction_file.replace('.jsonl', '_score.jsonl')
    if os.path.exists(score_output_file):
        df = pd.read_json(score_output_file, lines=True)
    else:
        thresholds = np.linspace(0.001, 1, 1000)
        threshold_candidate = 0
        max_score = {'position_score': 0}
        all_gold_answers, all_questions = get_all_gold_answers(args.gold_file)
        df = pd.read_json(args.prediction_file, lines=True)

        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(task, 
                repeat(df),
                list(thresholds),
                repeat(all_gold_answers),
                repeat(all_questions),
                repeat(args.limit_num_wrong_answers)),
                total=len(thresholds))
            )
        df = pd.DataFrame(results)
        df.to_json(args.prediction_file.replace('.jsonl', '_score.jsonl'), orient='records', lines=True, force_ascii=False)
    
    print(df)
    
    thresholds = df['confidence_threshold'].values
    accuracy_score = df['accuracy_score'].values
    position_score = df['position_score'].values
    total_score = df['total_score'].values

    accuracy_max_idx = df['accuracy_score'].idxmax()
    position_max_idx = df['position_score'].idxmax()
    total_max_idx = df['total_score'].idxmax()

    colors = ['red', 'green', 'blue']
    plt.plot(thresholds, accuracy_score, label='accuracy_score', color=colors[0])
    plt.text(x=thresholds[accuracy_max_idx], y=accuracy_score[accuracy_max_idx], s=f"({thresholds[accuracy_max_idx]:.3f}, {accuracy_score[accuracy_max_idx]:.3f})", color=colors[0])

    plt.plot(thresholds, position_score, label='position_score', color=colors[1])
    plt.text(x=thresholds[position_max_idx], y=position_score[position_max_idx], s=f"({thresholds[position_max_idx]:.3f}, {position_score[position_max_idx]:.3f})", color=colors[1])

    plt.plot(thresholds, total_score, label='total_score', color=colors[2])
    plt.text(x=thresholds[total_max_idx], y=total_score[total_max_idx], s=f"({thresholds[total_max_idx]:.3f}, {total_score[total_max_idx]:.3f})", color=colors[2])

    plt.xlabel('confidence_threshold')
    plt.ylabel('score')
    plt.legend()
    plt.savefig(args.prediction_file.replace('.jsonl', '_score.png'))

    report_outline = [
        ('accuracy_score', accuracy_max_idx),
        ('position_score', position_max_idx),
        ('total_score', total_max_idx),
    ]

    for label, confidence in report_outline:
        print("\n------max_" + label + "-----")
        print(df.iloc[confidence])