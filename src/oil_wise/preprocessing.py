"""Korean news preprocessing with a dependency-light fallback tokenizer."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z]{2,}")
DEFAULT_STOPWORDS = {
    "기자",
    "뉴스",
    "연합뉴스",
    "서울",
    "대한",
    "관련",
    "통해",
    "위해",
    "이번",
    "지난",
    "오는",
    "하는",
    "했다",
    "있다",
    "있는",
    "대한민국",
}


@dataclass(slots=True)
class KoreanNewsTokenizer:
    """Tokenize Korean news with Kiwi when available and regex otherwise."""

    stopwords: set[str] = field(default_factory=lambda: set(DEFAULT_STOPWORDS))
    use_kiwi: bool = True
    allowed_tags: frozenset[str] = frozenset({"NNG", "NNP", "SL"})
    _kiwi: object | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.use_kiwi:
            try:
                from kiwipiepy import Kiwi

                self._kiwi = Kiwi()
            except ImportError:
                self._kiwi = None

    @property
    def backend(self) -> str:
        return "kiwi" if self._kiwi is not None else "regex"

    def __call__(self, text: str) -> list[str]:
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", str(text))
        if self._kiwi is not None:
            tokens = [
                token.form.lower()
                for token in self._kiwi.tokenize(cleaned)
                if token.tag in self.allowed_tags
            ]
        else:
            tokens = [match.group(0).lower() for match in TOKEN_PATTERN.finditer(cleaned)]
        return [token for token in tokens if token not in self.stopwords and len(token) >= 2]

    def transform(self, documents: Iterable[str]) -> list[list[str]]:
        return [self(document) for document in documents]
