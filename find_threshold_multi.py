from compute_score_new import compute_scores, get_all_gold_answers, get_all_pred_answers
import clean_results
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from matplotlib import pyplot as plt

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

    results = {scores['confidence_threshold']: scores for scores in results}
    confidence = []
    accuracy_score = []
    position_score = []
    total_score = []
    for _, scores in results.items():
        confidence.append(scores['confidence_threshold'])
        accuracy_score.append(scores['accuracy_score'])
        position_score.append(scores['position_score'])
        total_score.append(scores['total_score'])

    colors = ['red', 'green', 'blue']
    plt.plot(thresholds, accuracy_score, label='accuracy_score', color=colors[0])
    plt.text(x=thresholds[np.argmax(accuracy_score)], y=np.max(accuracy_score), s=f"({thresholds[np.argmax(accuracy_score)]:.3f}, {np.max(accuracy_score):.3f})", color=colors[0])

    plt.plot(thresholds, position_score, label='position_score', color=colors[1])
    plt.text(x=thresholds[np.argmax(position_score)], y=np.max(position_score), s=f"({thresholds[np.argmax(position_score)]:.3f}, {np.max(position_score):.3f})", color=colors[1])

    plt.plot(thresholds, total_score, label='total_score')
    plt.text(x=thresholds[np.argmax(total_score)], y=np.max(total_score), s=f"({thresholds[np.argmax(total_score)]:.3f}, {np.max(total_score):.3f})", color=colors[2])

    plt.xlabel('confidence_threshold')
    plt.ylabel('score')
    plt.savefig(args.prediction_file.replace('.jsonl', '_score.png'))

    
    report_outline = [
        ('accuracy_score', thresholds[np.argmax(accuracy_score)]),
        ('position_score', thresholds[np.argmax(position_score)]),
        ('total_score', thresholds[np.argmax(total_score)]),
    ]

    for label, confidence in report_outline:
        print("\n------max_" + label + "-----")
        scores = results[confidence]
        print("confidence_threshold: {}".format(scores["confidence_threshold"]))
        print("num_questions: {}".format(scores["num_questions"]))
        print("num_correct: {}".format(scores["num_correct"]))
        print("num_missed: {}".format(scores["num_missed"]))
        print("num_failed: {}".format(scores["num_failed"]))
        print("accuracy: {:.1%}".format(scores["accuracy"]))
        print("accuracy_score: {:.3f}".format(scores["accuracy_score"]))
        print("position_score: {:.3f}".format(scores["position_score"]))
        print("total_score: {:.3f}".format(scores["total_score"]))