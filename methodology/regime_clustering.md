# Regime Clustering Methodology

The regime analysis uses unsupervised clustering to determine whether flow behaves differently under distinct market states.

## State Features
Market state is described using features such as:

- flow intensity
- spread
- displayed depth
- book imbalance
- realized return volatility
- realized variance
- moneyness
- option type encoding

These variables are intended to describe the environment in which flow is observed, not the outcome itself.

## Why Unsupervised Learning
The purpose is discovery rather than prediction. Clustering allows the study to partition the dataset into interpretable groups without forcing a label or hand-crafted regime definition.

## Cluster Interpretation
After clustering, each cluster is evaluated using residual flow correlations. A useful regime would be one in which mean residual correlation is consistently and materially positive. In this study, the clusters remain broadly weak, suggesting that state segmentation alone does not recover a robust flow-driven edge.
