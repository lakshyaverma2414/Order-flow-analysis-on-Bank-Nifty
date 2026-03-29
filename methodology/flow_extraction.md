# Flow Extraction Methodology

The objective is to reconstruct a continuous order-flow signal from irregular option ticks.

## Signed Flow Reconstruction
Signed trade flow is inferred from the relationship between trade price and the best quotes:

- if trade price is at or above the best ask, the trade is treated as buyer-initiated
- if trade price is at or below the best bid, the trade is treated as seller-initiated
- otherwise the trade is treated as neutral

This produces signed traded quantity in event time.

## Event-Time Normalization
Because option ticks arrive irregularly, all flow measures are computed in event time using the elapsed seconds between consecutive ticks. Time-aware exponential weighting is used so that long pauses and fast bursts are treated consistently.

## Core Continuous Features
The main flow descriptors are:

- intensity: smoothed absolute signed volume per unit time
- imbalance: smoothed signed flow relative to total flow magnitude
- absorption: inferred pressure from changes in top-of-book depth

These are combined into a continuous flow signal rather than a threshold-based classifier.
