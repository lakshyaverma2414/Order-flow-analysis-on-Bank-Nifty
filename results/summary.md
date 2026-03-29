# Results Summary

This folder contains the integrated empirical outputs used throughout the repository:

- [flow_global_summary.json](flow_global_summary.json)
- [cluster_summary_v1.json](cluster_summary_v1.json)
- [cluster_summary_v2.json](cluster_summary_v2.json)
- `feature_dataset.csv`
- `regime_dataset_v2.csv`

## Full-Sample Result
Across **27 trading days** and **3135 usable strike-day observations**, the mean residual Pearson correlation is **-0.0010** with standard deviation **0.0749**. The 5th and 95th percentiles are **-0.1093** and **0.1130**, respectively. This is a broad but symmetric cross-section centered near zero rather than a distribution shifted into persistent positive territory.

## Statistical Insignificance
The estimated standard error of the cross-file mean is **0.0013**, producing a t-statistic of **-0.76**. That magnitude is too small to distinguish the sample mean from noise. The implied signal-to-noise ratio is **0.0137**, reinforcing the same conclusion from another angle: the residual signal is not merely weak, it is effectively absent at the aggregate level.

## Clustering Interpretation
### V1
The first clustering pass produced cluster-level mean residual correlations ranging from **-0.0100** to **0.0101**. Those values remain close to zero even before correcting the methodological issue that V1 allowed outcome-linked variables into the clustering stage.

### V2
The corrected state-only clustering pass narrowed the range further, from **-0.0049** to **0.0011**. This matters because it rules out the easy counterargument that "alpha only exists in certain market states." The data does not support that view here.

## Practical Reading
The integrated evidence is consistent across summary statistics, distributional views, and cluster analysis:

- spot-adjusted order flow does not show stable predictive correlation
- cluster structure exists as a description of market state, but not as a source of predictive regimes
- the most defensible interpretation is that observed option flow is reactive to market moves rather than a leading indicator of future residual premium change
