# Data contract

Oil-Wise AI deliberately keeps raw market and news data out of Git. This
document defines the minimum input schema expected by the command-line
workflow.

## WTI price CSV

Required semantic fields:

| Semantic field | Recognized examples | Type |
|---|---|---|
| Trading date | `date`, `datetime`, `날짜`, `일자` | parseable datetime |
| Closing price | `close`, `price`, `wti`, `종가`, `가격` | numeric |

The loader removes commas and dollar signs from price strings, normalizes
timestamps to dates, drops invalid rows, and keeps the last row when a date is
duplicated.

```csv
date,close
2024-01-02,70.38
2024-01-03,72.70
```

## Korean news CSV

Required semantic fields:

| Semantic field | Recognized examples | Type |
|---|---|---|
| Published time | `date`, `published_at`, `작성일`, `등록일`, `날짜` | parseable datetime |
| Title | `title`, `headline`, `제목`, `기사제목` | text |
| Body | `content`, `body`, `article`, `본문`, `기사내용` | text |

At least one title or body field is required. When both are present they are
combined. Empty documents and invalid dates are removed. Articles published on
the same date are concatenated into a daily document and retain an
`article_count`.

```csv
date,title,content
2024-01-02 09:00:00,OPEC 감산 논의,공급 감소 가능성이 제기됐다.
```

## Daily topic probability CSV

The `oil-wise topics` command emits:

```csv
date,topic_0,topic_1,topic_2,topic_3,topic_4,topic_5,topic_6,article_count
2024-01-02,0.10,0.15,0.05,0.20,0.10,0.15,0.25,12
```

Topic probabilities are joined to WTI trading days by normalized date. A
trading day without matched news receives zero topic features. Future news is
never backfilled into earlier trading days.

## Leakage controls

- Rows are sorted chronologically before splitting.
- Feature and target scalers are fit only on the training rows.
- A sample's target is exactly `sequence_length + forecast_horizon - 1`
  positions after the start of its input sequence.
- Samples are assigned to train/test by target date.
- Test metrics are calculated after inverse transformation to USD.

## Data governance

Before using a real dataset, verify:

- the provider allows local processing and derived research outputs;
- redistribution of full article text is permitted;
- personally identifiable information is removed when applicable;
- publication timestamps and market timestamps share an explicit timezone;
- revised prices and duplicate articles are handled consistently.
