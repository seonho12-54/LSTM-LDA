"""Project defaults derived from the Oil-Wise AI experiment."""

from __future__ import annotations

from dataclasses import dataclass

TOPIC_LABELS = {
    0: "정부 정책",
    1: "수출 및 경제",
    2: "에너지 및 환경",
    3: "산업 기술",
    4: "공공 안전",
    5: "외교 및 국제 협력",
    6: "국제 정세 및 경제 영향",
}

TOPIC_SEEDS = {
    0: ["유가", "가격", "시장", "거래", "변동", "경제", "지수", "추세", "투자", "분석"],
    1: ["공급", "수요", "생산", "소비", "재고", "수출", "수입", "가격", "원유", "석유"],
    2: ["분쟁", "제재", "지정학", "충돌", "러시아", "미국", "중국", "무역", "협상", "안보"],
    3: ["성장", "인플레이션", "불황", "고용", "금리", "정부", "재정", "화폐", "산업", "경기"],
    4: ["재생", "전환", "친환경", "탄소", "배출", "지속가능", "대체", "발전", "효율", "혁신"],
    5: ["기술", "프레임", "자동화", "인공지능", "데이터", "탐사", "미래", "연구", "정제", "운송"],
    6: ["환경", "기후", "오염", "규제", "보호", "재활용", "생태", "녹색", "청정", "자연"],
}


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Reproducible settings for topic modelling and LSTM experiments."""

    num_topics: int = 7
    lda_passes: int = 30
    lda_iterations: int = 100
    random_seed: int = 42
    sequence_length: int = 20
    forecast_horizon: int = 5
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 200

    def __post_init__(self) -> None:
        if self.num_topics < 2:
            raise ValueError("num_topics must be at least 2")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.forecast_horizon < 1:
            raise ValueError("forecast_horizon must be positive")
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.epochs < 1:
            raise ValueError("learning_rate and epochs must be positive")
