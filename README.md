# GPT zero-shot baseline


## 再現
### 環境構築
```
source setup.sh
```

### 確信度しきい値計算
```
python main.py model=rinna-1b dataset=aio2_v1.0
```

#### 結果
| model | threshold |
| -  | -- |
| rinna-1b| 0.839 |


### リーダーボード用データで推論

```
python main.py model=rinna-1b model.confidence_threshold=0.8739135913591359
```

主な引数選択肢
| 引数名                     | 型    | 選択肢                               |
| -------------------------- | ----- | ------------------------------------ |
| model                      | str   | rinna-1b (default) <br> open-calm-1b |
| model.confidence_threshold | float |                                    |
| dataset                    | str   | aio4_v1.0 (default) <br> aio2_v1.0   |


### しきい値を変えた結果を出力
例: 
```
python clean_results.py --prediction_file outputs/2023-08-31/12-31-02/dev_unlabeled.jsonl --confidence_threshold 0.8739135913591359
```