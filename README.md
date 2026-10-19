# Word Bigram Language Model

## 兩個程式

1. `build_tables.py`：使用 Jieba 斷詞，建立 1%、10% 或 100% 的 SQLite 資料表。
2. `main.py`：只開啟已完成的 SQLite 資料表，提供 `predict`、`prob`、`generate` 三項功能，不會重新讀取語料。

首次執行前，將 `wiki_article_list_2023_tra.json` 放在本資料夾，並安裝套件：

```powershell
py -3 -m pip install -r requirements.txt
```

## 建立資料表

```powershell
py -3 .\build_tables.py
```

Windows 也可以雙擊 `build_tables.bat`。預設建立 1%、10%、100% 三張表，依語料文章原始順序取前 N 篇；100% 表使用 `wiki_bigram_model.sqlite3`。標點與數字會保留為 token，句末標點也會計入 unigram 與 bigram。建表會先計算文章數，再顯示進度；每張表完成後會把 N、V、相異 bigram 數與耗時追加到 `build_metrics.csv`。若目標資料庫已存在，程式會先詢問是否重建。

## 使用三項功能

```powershell
py -3 .\main.py
```

或雙擊 `run.bat`。選單提供：

- `predict(word, k)`：查詢某詞最常接續的詞及機率。
- `prob(sentence)`：以 Jieba 切分並顯示輸入句子，再計算 bigram log 機率。
- `generate(n)`：選擇最高機率後續詞或加權抽樣，產生 n 個詞並直接串接顯示；最高機率模式隨機抽句首、後續取最高頻詞，加權抽樣則每一步都依頻率抽樣；兩種模式都可用整數 seed 重現結果。

啟動後可選擇 1%、10% 或 100% 資料表。查詢入口不會載入整份語料；若所選資料表不存在，請先執行 `build_tables.py` 建立。

## 比較 Jieba 與 CKIP

若尚未安裝 CKIP Transformers，先執行：

```powershell
py -3 -m pip install ckip-transformers
```

再啟動比較程式：

```powershell
py -3 .\compare_segmenters.py
```

可連續輸入句子查看兩種斷詞結果，輸入 `q` 離開。CKIP 使用 BERT-tiny CPU 模型；第一次執行會下載模型。
