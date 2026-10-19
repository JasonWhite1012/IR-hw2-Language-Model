from pathlib import Path
import re
from tempfile import TemporaryDirectory

from bigram_model import BigramModel, build_database, open_query_database


class FakeSegmenter:
    def __call__(self, sentences, use_delim=False):
        return [
            re.findall(r"[^\s。！？!?；;，、：:]+|[。！？!?；;，、：:]", sentence)
            for sentence in sentences
        ]


def test_build_and_query_small_json_array():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "small.json"
        database = root / "model.sqlite3"
        corpus.write_text('[\n"我 喜歡 貓。",\n"你 喜歡 狗。"\n]\n', encoding="utf-8")

        import bigram_model

        original_factory = bigram_model.create_segmenter
        bigram_model.create_segmenter = FakeSegmenter
        try:
            stats = build_database(corpus, database)
            sample_stats = build_database(
                corpus,
                root / "sample.sqlite3",
                article_limit=1,
                corpus_articles=2,
                sample_percent=50,
            )
        finally:
            bigram_model.create_segmenter = original_factory

        assert stats == (8, 6, 9)
        assert sample_stats == (4, 4, 5)
        connection = open_query_database(database)
        try:
            model = BigramModel(connection, FakeSegmenter())
            assert model.predict("我", 2) == [("喜歡", 1.0)]
            assert model.probability("我 喜歡 貓") == float("-inf")
            assert model.probability("我 喜歡 貓。") > float("-inf")
            sampled = model.generate(12, mode="sample", seed=7)
            assert len(sampled) == 12
            assert sampled == model.generate(12, mode="sample", seed=7)
            assert set(sampled) <= {"我", "你", "喜歡", "貓", "狗", "。"}
            progress = []
            greedy = model.generate(
                12,
                mode="greedy",
                seed=7,
                progress_callback=lambda generated, total: progress.append((generated, total)),
            )
            assert len(greedy) == 12
            assert greedy == model.generate(12, mode="greedy", seed=7)
            other_seed_greedy = model.generate(12, mode="greedy", seed=999)
            assert greedy[0] != other_seed_greedy[0]
            assert greedy[1] == other_seed_greedy[1] == "喜歡"
            assert progress[-1] == (12, 12)
        finally:
            connection.close()


def test_build_retains_punctuation_tokens():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "punctuation.json"
        database = root / "model.sqlite3"
        corpus.write_text('[\n"甲 ， 乙。"\n]\n', encoding="utf-8")

        import bigram_model

        original_factory = bigram_model.create_segmenter
        bigram_model.create_segmenter = FakeSegmenter
        try:
            stats = build_database(corpus, database)
        finally:
            bigram_model.create_segmenter = original_factory

        assert stats == (4, 4, 5)
        connection = open_query_database(database)
        try:
            words = {
                row[0]
                for row in connection.execute("SELECT word FROM unigrams")
            }
            assert words == {"甲", "乙", "，", "。"}
            assert connection.execute(
                "SELECT count FROM bigrams WHERE left_word = '甲' AND right_word = '，'"
            ).fetchone() == (1,)
            model = BigramModel(connection, FakeSegmenter())
            assert model.probability("甲 ， 乙。") > float("-inf")
            assert model.probability("甲 乙。") == float("-inf")
        finally:
            connection.close()
