from __future__ import annotations

import sqlite3

from bigram_model import (
    BigramModel,
    create_segmenter,
    database_path_for_percent,
    open_query_database,
)


def read_word(model: BigramModel, prompt: str) -> str:
    text = input(prompt).strip()
    tokens = model.tokenize(text)
    if not tokens:
        return ""
    if len(tokens) > 1:
        print(f"輸入被 Jieba 切成多個詞，使用第一個詞：{tokens[0]}")
    return tokens[0]


def run_menu(model: BigramModel) -> None:
    while True:
        print("\n=== Word Bigram Language Model ===")
        print("1. predict(word, k)")
        print("2. prob(sentence)")
        print("3. generate(seed, n)")
        print("4. 離開")
        choice = input("請選擇功能 (1-4): ").strip()

        if choice == "1":
            word = read_word(model, "輸入要預測的詞：")
            if not word:
                print("沒有取得有效詞語。")
                continue
            raw_k = input("top-k (預設 5): ").strip() or "5"
            try:
                predictions = model.predict(word, top_k=max(0, int(raw_k)))
            except ValueError:
                print("top-k 必須是整數。")
                continue
            if not predictions:
                print("資料表中沒有這個詞的後續 bigram。")
            for next_word, probability in predictions:
                print(f"{word} -> {next_word}: {probability:.6f}")

        elif choice == "2":
            sentence = input("輸入句子：").strip()
            print(f"Jieba 切詞：{model.tokenize(sentence)}")
            print(f"log P(sentence) = {model.probability(sentence)}")

        elif choice == "3":
            print("生成模式：1. 最高機率後續詞  2. 加權抽樣")
            raw_mode = input("請選擇模式 (1-2): ").strip()
            if raw_mode not in {"1", "2"}:
                print("無效模式，請輸入 1 或 2。")
                continue
            mode = "greedy" if raw_mode == "1" else "sample"
            raw_seed = input("輸入隨機種子（整數）：").strip()
            try:
                seed = int(raw_seed)
            except ValueError:
                print("隨機種子必須是整數。")
                continue
            raw_length = input("生成詞數 (預設 10): ").strip() or "10"
            try:
                length = max(0, int(raw_length))
            except ValueError:
                print("生成詞數必須是整數。")
                continue

            def show_progress(generated: int, total: int) -> None:
                print(f"\r生成進度：{generated}/{total} 詞", end="", flush=True)

            generated = model.generate(
                length,
                mode=mode,
                seed=seed,
                progress_callback=show_progress,
            )
            if length > 0:
                print()
            print("".join(generated))

        elif choice in {"4", "q", "quit", "exit"}:
            break
        else:
            print("無效選項，請輸入 1、2、3 或 4。")


def main() -> int:
    connection = None
    try:
        print("選擇語料比例：1. 1%  2. 10%  3. 100%")
        selection = input("請選擇資料表 (1-3): ").strip()
        percentages = {"1": 1, "2": 10, "3": 100}
        if selection not in percentages:
            print("無效選項，請輸入 1、2 或 3。")
            return 1
        database_path = database_path_for_percent(percentages[selection])
        connection = open_query_database(database_path)
        segmenter = create_segmenter()
    except (FileNotFoundError, RuntimeError, sqlite3.Error, ValueError) as error:
        if connection is not None:
            connection.close()
        print(f"無法啟動：{error}")
        return 1

    try:
        print(f"已載入資料表：{database_path}")
        run_menu(BigramModel(connection, segmenter))
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
