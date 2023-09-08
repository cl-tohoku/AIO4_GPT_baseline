import argparse
import pandas as pd
from tqdm import tqdm


def main(args: argparse.Namespace):
    assert args.original_path.endswith('.jsonl')
    df_original = pd.read_json(args.original_path, lines=True)
    columns = [
        'qid',
        'question',
        'answers'
    ]
    df_original = df_original[columns]
    df_original = df_original.sample(n=args.n_samples, random_state=42).sort_index()
    df_original.to_json(args.original_path.replace('.jsonl', f'_{args.n_samples}.jsonl'), orient='records', lines=True, force_ascii=False)


    df_data = []
    for _, row in tqdm(df_original.iterrows()):
        qid = row['qid']
        question = row['question']
        answers = row['answers']
        for i in range(len(question)):
            position = i + 1
            df_data.append({
                'qid': qid,
                'position': position,
                'question': question[:position],
                'answers': answers
            })

    df = pd.DataFrame(df_data)
    
    print(df.shape)

    df['index'] = df.index

    df_complete_question = df.groupby('qid').last()

    def format_question(row):
        if row['index'] in df_complete_question['index'].values:
            return row['question'] + ' 答えは「'
        else:
            return row['question'] + '...? 答えは「'
        
    df['question'] = df.apply(format_question, axis=1)
    df = df.drop(columns=['index'])
    df.to_json(args.original_path.replace('.jsonl', f'_{args.n_samples}_processed.jsonl'), orient='records', lines=True, force_ascii=False)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, required=True)

    args = parser.parse_args()
    main(args)
    