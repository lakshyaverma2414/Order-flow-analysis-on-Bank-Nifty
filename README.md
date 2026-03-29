# Flow Microstructure Study on Bank Nifty Index

## 1. Title
Large-Scale Empirical Study of Order Flow Predictability in **BankNifty** Index Options

## 2. Motivation
Order flow is often described as an informative microstructure variable in listed derivatives. The BankNifty options market provides a high-liquidity environment where order flow is often assumed to carry informational content. That claim is easy to overstate when raw premium changes are heavily contaminated by movements in the underlying index. This repository documents a negative-result research study: we reconstruct continuous event-time option flow, remove spot-driven premium effects, and test whether any residual predictive signal remains at scale. The objective is not to build a trading system. It is to answer a narrower empirical question with discipline: does reconstructed order flow lead future option premium changes once the dominant spot component is removed?

## 3. Dataset
The study uses irregular tick-level **BankNifty index options (NSE India)** data with full-day files for individual strikes. Each file contains timestamped trades, top-of-book prices and quantities, and synchronized spot prices. The current sample spans **27 trading days**, **3174 processed strike-day files**, and **3135 usable residual-correlation observations** after data-quality filtering. This is a high-liquidity index derivatives market, which makes it a useful setting for testing whether apparent flow information survives careful spot adjustment.

### Data Organization
The dataset is structured as an event-time warehouse designed for large-scale tick processing:

- hierarchical layout by month and trading day
- per-file storage at strike-day granularity
- preservation of irregular event-time spacing without resampling
- efficient traversal across 3000+ strike-day files

## 4. Methodology
The research pipeline for BankNifty option ticks proceeds in four steps:

1. **Flow reconstruction**: infer signed order flow from trade price relative to prevailing bid and ask quotes, then smooth it in event time using time-aware exponential updates.
2. **Target construction**: measure future option premium change over a fixed forward horizon using the first future tick at or beyond the horizon.
3. **Spot adjustment**: regress future premium change on the option-type-adjusted spot move and retain the residual premium move as the quantity of interest.
4. **Regime discovery**: cluster observations by market state features to test whether any latent state preserves residual predictive power.

Methodological details are documented in [methodology/flow_extraction.md](methodology/flow_extraction.md), [methodology/spot_adjustment.md](methodology/spot_adjustment.md), and [methodology/regime_clustering.md](methodology/regime_clustering.md).

## 5. Results
The headline result is straightforward: **after spot adjustment, the average predictive signal is approximately zero**. The full-sample residual Pearson correlation distribution, stored in [results/flow_global_summary.json](results/flow_global_summary.json), has:

- mean `pearson_residual` = **-0.0010**
- standard deviation = **0.075**
- median = **-0.0036**
- 5th percentile = **-0.11**
- 95th percentile = **0.11**

These numbers imply a distribution centered almost exactly at zero, with ordinary cross-file variation but no persistent positive drift. The residual Spearman correlation mean is also close to zero at **-0.0088**.

## 6. Statistical Significance
The sample mean is economically and statistically negligible. Using the cross-file mean and standard deviation from [results/flow_global_summary.json](results/flow_global_summary.json), the standard error of the mean is approximately **0.0013**, which yields a t-statistic of **-0.76**. In other words, the estimated signal is indistinguishable from noise at conventional significance levels. The signal-to-noise ratio, measured as `|mean| / std`, is only **0.0137**. This is not a weak edge that needs more polishing; it is a distribution whose center is effectively null.

## 7. Methodological Iteration (V1 -> V2)
Two clustering passes were used to test whether the null result was an artifact of regime misspecification.

- **V1** clustered partially on realized output-side variables, including residual rank structure. That approach was useful diagnostically but methodologically impure because it allowed the clustering stage to see part of the outcome landscape. Even under that favorable setup, cluster means in [results/cluster_summary_v1.json](results/cluster_summary_v1.json) remained confined to roughly **-0.0100** to **0.0101**.
- **V2** corrected the design by clustering only on market-state features such as flow intensity, spread, depth, realized variance, moneyness, and option type. The resulting summaries in [results/cluster_summary_v2.json](results/cluster_summary_v2.json) tightened further, with cluster-level mean residual correlations between **-0.0049** and **0.0011**.

The important point is not that V2 looks cleaner. It is that the conclusion survives the cleaner design: **no predictive regime emerges even after clustering on state variables rather than outcomes**.

## 8. Interpretation
The empirical picture is more consistent with **reactive** flow than **leading** flow. Three mechanisms are natural candidates:

- **Dealer hedging**: option flow responds mechanically to spot and volatility changes rather than forecasting them.
- **Post-move execution**: trades occur after the underlying move has already started, making the recorded flow contemporaneous or lagging.
- **Liquidity adjustment**: top-of-book changes and trade prints reflect inventory management and spread control, not latent directional information.

Taken together, the evidence suggests that option flow in this sample **follows price formation more than it precedes it**.

## 9. Implications
This repository is a research artifact built around hypothesis rejection. The results argue against treating reconstructed standalone option flow as a reliable alpha source after controlling for spot exposure. Flow may still be useful for descriptive market-state tagging, execution-quality analysis, or inventory pressure studies, but the evidence here does not support the claim that it systematically predicts future residual premium changes. These findings are particularly relevant for high-liquidity index derivatives such as BankNifty, where order flow is heavily influenced by hedging and liquidity provision dynamics.

## 10. Reproducibility
The repository is organized so the negative result can be reproduced transparently:

- raw-tick schema is documented in [data_format/schema.md](data_format/schema.md)
- methodology notes describe flow extraction, spot adjustment, and clustering
- integrated result artifacts live in [results/flow_global_summary.json](results/flow_global_summary.json), [results/cluster_summary_v1.json](results/cluster_summary_v1.json), and [results/cluster_summary_v2.json](results/cluster_summary_v2.json)
- per-file datasets used for visualization live in `results/feature_dataset.csv` and `results/regime_dataset_v2.csv`
- plots can be regenerated with [scripts/generate_plots.py](scripts/generate_plots.py)

The goal of the project is clarity, not persuasion. The central finding is a null: once spot-driven effects are removed, reconstructed order flow in this dataset does not exhibit reliable predictive power over future option premium changes.
