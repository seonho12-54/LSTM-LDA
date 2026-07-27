# Oil-Wise AI poster

The original one-page project poster is preserved in this repository:

- [Open the full PDF](./oil-wise-poster.pdf)
- [Open the optimized PNG preview](./assets/oil-wise-poster.png)

## Poster summary

The poster describes a two-stage oil-price forecasting experiment:

1. Korean news is converted to seven daily topic-probability signals with
   Seeded LDA.
2. Those topic signals are fed into a multivariate PyTorch LSTM to predict
   future WTI closing prices.
3. Sequence lengths and forecast horizons of 5, 10, 20, and 30 trading days
   are compared with MAE and RMSE.

The highlighted poster result uses a 20-day input sequence and a 5-day
forecast horizon, reporting MAE 6.1554 and RMSE 7.4439.

## Reproducibility note

The poster records the original experiment. The package in `src/oil_wise/`
reimplements that architecture with explicit data contracts, chronological
splits, train-only scaling, reproducible seeds, tests, and command-line
artifacts. Results can vary with dataset revisions, preprocessing choices,
dependency versions, and random seeds.
