from oil_wise.preprocessing import KoreanNewsTokenizer


def test_regex_tokenizer_removes_urls_stopwords_and_single_characters() -> None:
    tokenizer = KoreanNewsTokenizer(use_kiwi=False, stopwords={"기자"})

    tokens = tokenizer("기자 WTI 유가 급등 A https://example.com 국제 시장")

    assert tokens == ["wti", "유가", "급등", "국제", "시장"]


def test_transform_preserves_document_count() -> None:
    tokenizer = KoreanNewsTokenizer(use_kiwi=False)

    result = tokenizer.transform(["원유 가격 상승", "OPEC 감산 결정"])

    assert len(result) == 2
    assert result[0] == ["원유", "가격", "상승"]
