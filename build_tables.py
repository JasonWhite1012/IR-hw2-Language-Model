from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

from bigram_model import (
    CORPUS_PATH,
    build_database,
    count_articles,
    database_path_for_percent,
)


def main() -> int:
    if not CORPUS_PATH.is_file():
        print(f"找不到語料檔：{CORPUS_PATH}")
        return 1

    print("建立資料表：1. 1%  2. 10%  3. 100%  4. 全部比例")
    selection = input("請選擇 (預設 4): ").strip() or "4"
    selections = {"1": [1], "2": [10], "3": [100], "4": [1, 10, 100]}
    if selection not in selections:
        print("無效選項，請輸入 1、2、3 或 4。")
        return 1

    print(f"語料：{CORPUS_PATH}")
    print("正在計算語料文章數...")
    try:
        total_articles = count_articles(CORPUS_PATH)
    except Exception as error:
        print(f"\n無法讀取語料：{error}")
        return 1
    if total_articles == 0:
        print("語料沒有文章，無法建立資料表。")
        return 1
    print(f"文章數：{total_articles}")

    overall_started = time.perf_counter()
    metrics_path = Path(__file__).with_name("build_metrics.csv")
    for sample_percent in selections[selection]:
        database_path = database_path_for_percent(sample_percent)
        if database_path.exists():
            answer = input(
                f"資料表已存在：{database_path}\n要重新建立嗎？(y/N): "
            ).strip().lower()
            if answer not in {"y", "yes"}:
                print(f"保留現有 {sample_percent}% 資料表。")
                continue

        article_limit = max(1, total_articles * sample_percent // 100)
        print(
            f"\n建立 {sample_percent}% 資料表：前 {article_limit} / "
            f"{total_articles} 篇文章"
        )
        started = time.perf_counter()

        def show_progress(percent: int) -> None:
            print(f"\r建表進度：{percent:3d}%", end="", flush=True)

        try:
            n, vocabulary, bigrams = build_database(
                database_path=database_path,
                progress_callback=show_progress,
                article_limit=article_limit,
                corpus_articles=total_articles,
                sample_percent=sample_percent,
            )
        except Exception as error:
            print(f"\n{sample_percent}% 建表失敗：{error}")
            return 1

        elapsed = time.perf_counter() - started
        print(
            f"\n{sample_percent}% 建表完成：N={n}, V={vocabulary}, B={bigrams}"
        )
        print(f"耗時：{elapsed:.1f} 秒")
        write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
        with metrics_path.open("a", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            if write_header:
                writer.writerow(
                    [
                        "built_at",
                        "sample_percent",
                        "sample_articles",
                        "corpus_articles",
                        "N",
                        "V",
                        "distinct_bigrams",
                        "build_seconds",
                        "database",
                    ]
                )
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    sample_percent,
                    article_limit,
                    total_articles,
                    n,
                    vocabulary,
                    bigrams,
                    f"{elapsed:.1f}",
                    database_path.name,
                ]
            )

    print(f"\n全部完成，總耗時：{time.perf_counter() - overall_started:.1f} 秒")
    print(f"建表指標已追加至：{metrics_path}")
    print("之後執行 main.py，選擇資料表比例即可查詢。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
