from __future__ import annotations

import json
import logging
import math
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

CORPUS_PATH = Path(__file__).with_name("wiki_article_list_2023_tra.json")
DATABASE_PATH = Path(__file__).with_name("wiki_bigram_model.sqlite3")
DATABASE_PATHS = {
    1: DATABASE_PATH.with_name("wiki_bigram_model_1pct.sqlite3"),
    10: DATABASE_PATH.with_name("wiki_bigram_model_10pct.sqlite3"),
    100: DATABASE_PATH,
}
SENTENCES_PER_BATCH = 64


def database_path_for_percent(percent: int) -> Path:
    try:
        return DATABASE_PATHS[percent]
    except KeyError as error:
        raise ValueError("語料比例必須是 1、10 或 100") from error


def create_segmenter():
    try:
        import jieba
    except ImportError as error:
        raise RuntimeError(
            "找不到 jieba，請先執行：py -3 -m pip install -r requirements.txt"
        ) from error
    jieba.setLogLevel(logging.ERROR)

    def segment_batch(sentences: list[str], use_delim: bool = False) -> list[list[str]]:
        return [list(jieba.cut(sentence, cut_all=False)) for sentence in sentences]

    return segment_batch


def split_sentences(text: str) -> list[str]:
    parts = re.findall(r"[^。！？!?；;\n]+[。！？!?；;]*|[。！？!?；;]+", text)
    return [part.strip() for part in parts if part.strip()]


def iter_articles(path: Path) -> Iterable[tuple[str, int]]:
    """Stream article strings from the assignment's JSON array, one item per line."""
    with path.open("rb") as source:
        for line in source:
            value = line.strip()
            if not value or value in (b"[", b"]"):
                continue
            if value.endswith(b","):
                value = value[:-1].rstrip()
            article = json.loads(value)
            if not isinstance(article, str):
                raise ValueError("語料格式錯誤：每一筆文章必須是 JSON 字串")
            yield article, source.tell()


def count_articles(path: Path) -> int:
    with path.open("rb") as source:
        return sum(
            1
            for line in source
            if line.strip() and line.strip() not in (b"[", b"]")
        )


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE unigrams (
            word TEXT PRIMARY KEY,
            count INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE bigrams (
            left_word TEXT NOT NULL,
            right_word TEXT NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY (left_word, right_word)
        ) WITHOUT ROWID;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )


def _write_sentence_batch(
    connection: sqlite3.Connection,
    sentences: list[str],
    segmenter,
) -> None:
    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[tuple[str, str]] = Counter()

    for tokens in segmenter(sentences, use_delim=False):
        tokens = [token for token in tokens if token and not token.isspace()]
        if not tokens:
            continue
        unigram_counts.update(tokens)
        sequence = ["<s>", *tokens, "</s>"]
        bigram_counts.update(zip(sequence, sequence[1:]))

    connection.executemany(
        """INSERT INTO unigrams (word, count) VALUES (?, ?)
        ON CONFLICT(word) DO UPDATE SET count = count + excluded.count""",
        unigram_counts.items(),
    )
    connection.executemany(
        """INSERT INTO bigrams (left_word, right_word, count) VALUES (?, ?, ?)
        ON CONFLICT(left_word, right_word)
        DO UPDATE SET count = count + excluded.count""",
        ((left, right, count) for (left, right), count in bigram_counts.items()),
    )


def build_database(
    corpus_path: Path = CORPUS_PATH,
    database_path: Path = DATABASE_PATH,
    progress_callback: Callable[[int], None] | None = None,
    article_limit: int | None = None,
    corpus_articles: int | None = None,
    sample_percent: int = 100,
) -> tuple[int, int, int]:
    corpus_path = corpus_path.resolve()
    database_path = database_path.resolve()
    if corpus_articles is None:
        corpus_articles = count_articles(corpus_path)
    if article_limit is None:
        article_limit = corpus_articles
    if not 1 <= article_limit <= corpus_articles:
        raise ValueError("article_limit must be between 1 and corpus_articles")

    building_path = database_path.with_name(database_path.name + ".building")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if building_path.exists():
        building_path.unlink()

    segmenter = create_segmenter()
    connection = sqlite3.connect(building_path)
    connection.execute("PRAGMA cache_size = -32768")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    _create_schema(connection)

    last_percent = -1
    try:
        for article_number, (article, _) in enumerate(iter_articles(corpus_path), start=1):
            sentences = split_sentences(article)
            for start in range(0, len(sentences), SENTENCES_PER_BATCH):
                _write_sentence_batch(
                    connection,
                    sentences[start : start + SENTENCES_PER_BATCH],
                    segmenter,
                )

            if article_number % 1000 == 0:
                connection.commit()

            percent = min(100, int(article_number * 100 / article_limit))
            if progress_callback is not None and percent > last_percent:
                progress_callback(percent)
                last_percent = percent
            if article_number >= article_limit:
                break

        connection.execute(
            "INSERT INTO metadata (key, value) VALUES ('complete', '1')"
        )
        connection.execute(
            "INSERT INTO metadata (key, value) VALUES ('corpus_path', ?)",
            (str(corpus_path),),
        )
        connection.execute(
            "INSERT INTO metadata (key, value) VALUES ('corpus_size', ?)",
            (str(corpus_path.stat().st_size),),
        )
        connection.execute(
            "INSERT INTO metadata (key, value) VALUES ('corpus_articles', ?)",
            (str(corpus_articles),),
        )
        connection.execute(
            "INSERT INTO metadata (key, value) VALUES ('sample_articles', ?)",
            (str(article_limit),),
        )
        connection.execute(
            "INSERT INTO metadata (key, value) VALUES ('sample_percent', ?)",
            (str(sample_percent),),
        )
        connection.execute(
            "INSERT INTO metadata (key, value) VALUES ('tokenizer', 'jieba')"
        )
        connection.commit()
        stats = (
            int(connection.execute("SELECT COALESCE(SUM(count), 0) FROM unigrams").fetchone()[0]),
            int(connection.execute("SELECT COUNT(*) FROM unigrams").fetchone()[0]),
            int(connection.execute("SELECT COUNT(*) FROM bigrams").fetchone()[0]),
        )
    finally:
        connection.close()

    if progress_callback is not None and last_percent < 100:
        progress_callback(100)
    building_path.replace(database_path)
    return stats


def open_query_database(database_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    if not database_path.is_file():
        raise FileNotFoundError("找不到模型資料表，請先執行 build_tables.py")
    connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'complete'"
        ).fetchone()
        if row is None or row[0] != "1":
            raise ValueError("模型資料表尚未完成，請重新執行 build_tables.py")
        tokenizer_row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'tokenizer'"
        ).fetchone()
        if tokenizer_row is None or tokenizer_row[0] != "jieba":
            raise ValueError("資料表不是使用 Jieba 建立，請重新執行 build_tables.py")
    except Exception:
        connection.close()
        raise
    return connection


class BigramModel:
    def __init__(self, connection: sqlite3.Connection, segmenter) -> None:
        self.connection = connection
        self.segmenter = segmenter

    def tokenize(self, text: str) -> list[str]:
        return [
            token
            for token in self.segmenter([text], use_delim=False)[0]
            if token and not token.isspace()
        ]

    def predict(self, word: str, top_k: int = 5) -> list[tuple[str, float]]:
        total = self.connection.execute(
            "SELECT COALESCE(SUM(count), 0) FROM bigrams WHERE left_word = ?",
            (word,),
        ).fetchone()[0]
        if not total:
            return []
        rows = self.connection.execute(
            """SELECT right_word, count FROM bigrams WHERE left_word = ?
            ORDER BY count DESC, right_word LIMIT ?""",
            (word, max(0, top_k)),
        ).fetchall()
        return [(right, count / total) for right, count in rows]

    def probability(self, sentence: str) -> float:
        sentences = split_sentences(sentence)
        if not sentences:
            return 0.0
        score = 0.0
        for part in sentences:
            tokens = self.tokenize(part)
            if not tokens:
                continue
            sequence = ["<s>", *tokens, "</s>"]
            for left, right in zip(sequence, sequence[1:]):
                numerator, denominator = self.connection.execute(
                    """SELECT COALESCE(SUM(CASE WHEN right_word = ? THEN count ELSE 0 END), 0),
                    COALESCE(SUM(count), 0) FROM bigrams WHERE left_word = ?""",
                    (right, left),
                ).fetchone()
                if not numerator or not denominator:
                    return float("-inf")
                score += math.log(numerator / denominator)
        return score

    def generate(
        self,
        length: int,
        mode: str = "sample",
        seed: int = 0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[str]:
        if mode not in {"greedy", "sample"}:
            raise ValueError("mode must be 'greedy' or 'sample'")
        generator = random.Random(seed)
        result: list[str] = []
        current = "<s>"
        target_length = max(0, length)

        while len(result) < target_length:
            rows = self.connection.execute(
                """SELECT right_word, count FROM bigrams WHERE left_word = ?
                ORDER BY count DESC, right_word""",
                (current,),
            ).fetchall()
            if not rows:
                if current == "<s>":
                    break
                current = "<s>"
                continue

            if mode == "greedy" and current != "<s>":
                next_word = rows[0][0]
            else:
                next_word = generator.choices(
                    [word for word, _ in rows],
                    weights=[count for _, count in rows],
                    k=1,
                )[0]
            if next_word == "</s>":
                current = "<s>"
                continue
            result.append(next_word)
            current = next_word
            if progress_callback is not None:
                progress_callback(len(result), target_length)

        return result
