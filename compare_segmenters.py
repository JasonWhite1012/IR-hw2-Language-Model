from __future__ import annotations

import logging
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO


def create_ckip_segmenter():
    try:
        with (
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings(
                "ignore",
                message="You are sending unauthenticated requests to the HF Hub.*",
            )
            from ckip_transformers.nlp import CkipWordSegmenter

            word_segmenter = CkipWordSegmenter(model="bert-tiny", device=-1)
    except ImportError as error:
        raise RuntimeError(
            "找不到 CKIP Transformers，請執行：py -3 -m pip install ckip-transformers"
        ) from error

    def segment(sentence: str) -> list[str]:
        with (
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings(
                "ignore",
                message="You are sending unauthenticated requests to the HF Hub.*",
            )
            return word_segmenter([sentence], use_delim=False)[0]

    return segment


def main() -> int:
    try:
        import jieba
    except ImportError as error:
        print("找不到 jieba，請先執行：py -3 -m pip install -r requirements.txt")
        return 1

    jieba.setLogLevel(logging.ERROR)
    try:
        ckip_segment = create_ckip_segmenter()
    except RuntimeError as error:
        print(error)
        return 1

    print("輸入句子比較 Jieba 與 CKIP；輸入 q 離開。")
    while True:
        try:
            sentence = input("句子：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if sentence.lower() in {"q", "quit", "exit"}:
            break
        if not sentence:
            continue

        jieba_tokens = list(jieba.cut(sentence, cut_all=False))
        ckip_tokens = ckip_segment(sentence)
        print(f"Jieba：{' / '.join(jieba_tokens)}")
        print(f"CKIP：{' / '.join(ckip_tokens)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
