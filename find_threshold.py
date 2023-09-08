from compute_score_new import compute_scores, get_all_gold_answers, get_all_pred_answers
from clean_results import clean_results 
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor

def find_threshold(
    gold_file: str,
    prediction_file: str,
    limit_num_wrong_answers: int= None, 
):
    thresholds = np.linspace(0.001, 1, 1000)
    threshold_candidate = 0
    max_score = {'position_score': 0}
    all_gold_answers, all_questions = get_all_gold_answers(gold_file)
    df = pd.read_json(prediction_file, lines=True)

    def task(confidence_threshold):
        print(confidence_threshold)
        # start = time.time()
        df_tmp = clean_results(df, confidence_threshold)
        # end = time.time()
        # print(end - start)

        # start = time.time()
        all_pred_answers = get_all_pred_answers(df=df_tmp)
        # end = time.time()
        # print(end - start)

        scores = compute_scores(
            all_gold_answers,
            all_questions,
            all_pred_answers,
            limit_num_wrong_answers=limit_num_wrong_answers,
        )
        return scores
    
    print(list(thresholds))
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(task, list(thresholds))
    # print(results)

    # for confidence_threshold in tqdm(thresholds):
    #     # start = time.time()
    #     df_tmp = clean_results(df, confidence_threshold)
    #     # end = time.time()
    #     # print(end - start)

    #     # start = time.time()
    #     all_pred_answers = get_all_pred_answers(df=df_tmp)
    #     # end = time.time()
    #     # print(end - start)

    #     scores = compute_scores(
    #         all_gold_answers,
    #         all_questions,
    #         all_pred_answers,
    #         limit_num_wrong_answers=limit_num_wrong_answers,
    #     )
    #     if scores['position_score'] > max_score['position_score']:
    #         max_score = scores
    #         threshold_candidate = confidence_threshold

    return threshold_candidate, max_score
        # score = compute_scores(gold_file, prediction_file, limit_num_wrong_answers=limit_num_wrong_answers, confidence_threshold=confidence_threshold)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--limit_num_wrong_answers", type=int)
    args = parser.parse_args()

    final_threshold, scores = find_threshold(args.gold_file, args.prediction_file, limit_num_wrong_answers=args.limit_num_wrong_answers)
    print(final_threshold)
    print("num_questions: {}".format(scores["num_questions"]))
    print("num_correct: {}".format(scores["num_correct"]))
    print("num_missed: {}".format(scores["num_missed"]))
    print("num_failed: {}".format(scores["num_failed"]))
    print("accuracy: {:.1%}".format(scores["accuracy"]))
    print("accuracy_score: {:.3f}".format(scores["accuracy_score"]))
    print("position_score: {:.3f}".format(scores["position_score"]))
    print("total_score: {:.3f}".format(scores["total_score"]))