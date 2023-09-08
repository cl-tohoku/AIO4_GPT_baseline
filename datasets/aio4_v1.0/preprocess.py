import argparse
import pandas as pd


def main(args: argparse.Namespace):
    assert args.original_path.endswith('.jsonl')
    df = pd.read_json(args.original_path, lines=True)

    df['index'] = df.index

    df_complete_question = df.groupby('qid').last()

    def format_question(row):
        if row['index'] in df_complete_question['index'].values:
            return row['question'] + ' 答えは「'
        else:
            return row['question'] + '...? 答えは「'

    df['question'] = df.apply(format_question, axis=1)
    
    df = df.drop(columns=['index'])

    df.to_json(args.original_path.replace('.jsonl', '_processed.jsonl'), orient='records', lines=True, force_ascii=False)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
    