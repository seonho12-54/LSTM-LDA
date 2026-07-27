"""Seeded LDA topic signals for Korean oil-market news."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from oil_wise.config import TOPIC_SEEDS
from oil_wise.preprocessing import KoreanNewsTokenizer


def build_eta(
    token_to_id: Mapping[str, int],
    num_topics: int,
    seed_topics: Mapping[int, Sequence[str]],
    base_prior: float = 0.01,
    seed_prior: float = 10.0,
) -> np.ndarray:
    """Build a topic-word prior matrix that boosts domain seed words."""

    if num_topics < 1 or not token_to_id:
        raise ValueError("num_topics and vocabulary must be non-empty")
    if base_prior <= 0 or seed_prior <= base_prior:
        raise ValueError("seed_prior must be greater than a positive base_prior")

    eta = np.full((num_topics, len(token_to_id)), base_prior, dtype=np.float32)
    for topic_id, words in seed_topics.items():
        if not 0 <= topic_id < num_topics:
            raise ValueError(f"seed topic {topic_id} is outside 0..{num_topics - 1}")
        for word in words:
            token_id = token_to_id.get(word.lower())
            if token_id is not None:
                eta[topic_id, token_id] = seed_prior
    return eta


class SeededLDATopicModel:
    """Gensim LDA wrapper with reproducible topic-word seed priors."""

    def __init__(
        self,
        num_topics: int = 7,
        seed_topics: Mapping[int, Sequence[str]] | None = None,
        passes: int = 30,
        iterations: int = 100,
        random_state: int = 42,
        no_below: int = 2,
        no_above: float = 0.95,
        keep_n: int = 20_000,
        tokenizer: KoreanNewsTokenizer | None = None,
    ) -> None:
        self.num_topics = num_topics
        self.seed_topics = dict(seed_topics or TOPIC_SEEDS)
        self.passes = passes
        self.iterations = iterations
        self.random_state = random_state
        self.no_below = no_below
        self.no_above = no_above
        self.keep_n = keep_n
        self.tokenizer = tokenizer or KoreanNewsTokenizer()
        self.dictionary: object | None = None
        self.model: object | None = None

    def fit(self, documents: Sequence[str]) -> SeededLDATopicModel:
        """Fit the seeded topic model on news documents."""

        if len(documents) < 2:
            raise ValueError("at least two documents are required")
        try:
            from gensim import corpora
            from gensim.models import LdaModel
        except ImportError as exc:
            raise ImportError(
                "Seeded LDA requires optional dependencies: pip install -e .[topic]"
            ) from exc

        tokens = self.tokenizer.transform(documents)
        dictionary = corpora.Dictionary(tokens)
        dictionary.filter_extremes(
            no_below=self.no_below,
            no_above=self.no_above,
            keep_n=self.keep_n,
        )
        if len(dictionary) == 0:
            raise ValueError("all tokens were filtered; lower no_below or provide more documents")

        corpus = [dictionary.doc2bow(document) for document in tokens]
        eta = build_eta(
            dictionary.token2id,
            num_topics=self.num_topics,
            seed_topics=self.seed_topics,
        )
        self.dictionary = dictionary
        self.model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            passes=self.passes,
            iterations=self.iterations,
            alpha="auto",
            eta=eta,
            minimum_probability=0.0,
            eval_every=None,
        )
        return self

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """Return a dense topic-probability vector for every document."""

        self._require_fitted()
        tokenized = self.tokenizer.transform(documents)
        rows: list[np.ndarray] = []
        for document in tokenized:
            bow = self.dictionary.doc2bow(document)
            distribution = self.model.get_document_topics(bow, minimum_probability=0.0)
            row = np.zeros(self.num_topics, dtype=np.float32)
            for topic_id, probability in distribution:
                row[topic_id] = probability
            rows.append(row)
        return np.stack(rows) if rows else np.empty((0, self.num_topics), dtype=np.float32)

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)

    def transform_daily(self, daily_documents: pd.DataFrame) -> pd.DataFrame:
        """Attach topic probabilities to a daily document frame."""

        if not {"date", "text"}.issubset(daily_documents.columns):
            raise ValueError("daily_documents must contain date and text columns")
        probabilities = self.transform(daily_documents["text"].astype(str).tolist())
        topics = pd.DataFrame(
            probabilities,
            columns=[f"topic_{topic_id}" for topic_id in range(self.num_topics)],
        )
        topics.insert(0, "date", pd.to_datetime(daily_documents["date"]).to_numpy())
        if "article_count" in daily_documents.columns:
            topics["article_count"] = daily_documents["article_count"].to_numpy()
        return topics

    def top_keywords(self, top_n: int = 10) -> dict[int, list[tuple[str, float]]]:
        """Return the most probable words for each learned topic."""

        self._require_fitted()
        return {
            topic_id: self.model.show_topic(topic_id, topn=top_n)
            for topic_id in range(self.num_topics)
        }

    def _require_fitted(self) -> None:
        if self.dictionary is None or self.model is None:
            raise RuntimeError("fit the topic model before calling this method")
