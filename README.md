# GPT zero-shot baseline
## 1. 再現
作成者実行環境  
- Python 3.9.13  


### 1.1. 環境構築
1. pytorchを入れる
2. 他のライブラリを入れる
    ```
    pip install -r requirements.txt
    ```

### 1.2. 学習データで推論
ここでは最適な確信度のしきい値を求めることを目的とする

#### 1.2.1. データ用意
ここではAI王公式配布データセットの学習データversion2.0の一部を使用
1. データ取得
   ```
   wget https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/aio_02_train.jsonl -P datasets/aio2/
   ```
2. **4000件サンプリング**
   ```
   python datasets/aio2/preprocess.py --original_path datasets/aio2/aio_02_train.jsonl --n_samples 4000
   ```

#### 1.2.2. しきい値探し
1. 推論: 結果は outputs/{date}/{time}/train.jsonl に出力される
   ```
   python main.py model=rinna-1b dataset=aio2_v1.0
   ```
   ```
   python main.py model=open-calm-1b dataset=aio2_v1.0
   ```
2. しきい値探し
   ```
   python find_threshold.py --prediction_file outputs/{date}/{time}/train.jsonl --gold_file datasets/aio2/aio_02_train_4000.jsonl --limit_num_wrong_answers 3
   ```

### しきい値探し結果

- rinna-1b
  - <img src="outputs/2023-07-05/23-28-26/train_score.png" width=50%>
- open-calm-1b
  - <img src="outputs/2023-07-13/11-15-13/train_score.png" width=50%>
### 1.3. リーダーボード用データで推論
#### 1.3.1. データ用意
準備中

#### 1.3.2. 推論: 結果は outputs/{date}/{time}/dev_unlabeled_cleaned.jsonl に出力される
```
python main.py model=rinna-1b model.confidence_threshold=0.865
```

#### 1.3.4. 結果
| model | threshold | accuracy score | position score | total score|
| -  | -| -| -|  -- |
| rinna-1b| 0.865 | 166 | 15.348 | 181.348 |
| open-calm-1b| 0.781 | 80 | 2.149 | 82.149 |

##### 参考
| model | threshold | accuracy score | position score | total score|
| -  | -| -| -|  -- |
| rinna-1b| 0.811 | 145 | 29.599 | 174.599 |
| open-calm-1b| 0.625 | 29 | 7.736 | 36.736 |


## 2. 参考
### 2.1. 主な引数選択肢

| 引数名                     | 型    | 選択肢                               |
| -------------------------- | ----- | ------------------------------------ |
| model                      | str   | rinna-1b (default) <br> open-calm-1b |
| model.confidence_threshold | float |                                    |
| dataset                    | str   | aio4_v1.0 (default) <br> aio2_v1.0   |


### 2.2. しきい値を変えた結果を出力
例: 
```
python clean_results.py --prediction_file outputs/2023-08-31/12-31-02/dev_unlabeled.jsonl --confidence_threshold 0.811
```