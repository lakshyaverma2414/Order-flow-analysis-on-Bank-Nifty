# Statistical Analysis

The study tests whether reconstructed option order flow carries predictive content for **future residual premium change** after removing the contribution explained by spot movement. The answer in this sample is negative.

## Core Statistics
- sample size: **3135** strike-day observations
- mean residual Pearson correlation: **-0.0010**
- residual correlation standard deviation: **0.0749**
- standard error of the mean: **0.0013**
- t-statistic for mean different from zero: **-0.76**
- signal-to-noise ratio `|mean| / std`: **0.0137**

## Distribution Shape
The residual correlation distribution is broad enough to show ordinary cross-file variation, but it is centered close to zero:

- 5th percentile: **-0.1093**
- median: **-0.0036**
- 95th percentile: **0.1130**

The near symmetry of the tails around zero and the small displacement of the median both argue against a hidden positive center. A genuine predictive signal would typically show some combination of a shifted mean, a shifted median, or a materially one-sided upper tail. None of those features is present here.

## Cluster-Level Evidence
The clustering results reinforce the same conclusion.

- V1 cluster means range from **-0.0100** to **0.0101**
- V2 cluster means range from **-0.0049** to **0.0011**

Even the most favorable cluster averages remain small relative to cross-file dispersion. That is exactly what one would expect if cluster structure captures heterogeneity in market conditions without uncovering a true predictive regime.

## Interpretation
Statistically, the null hypothesis of no meaningful residual predictive effect is difficult to reject. The mean is too small, the t-statistic is too close to zero, and the cluster-level summaries do not reveal a concentrated pocket of positive performance. The most careful reading is therefore conservative: any apparent order-flow relationship in raw premiums is absorbed by contemporaneous spot sensitivity rather than surviving as independent predictive information.
