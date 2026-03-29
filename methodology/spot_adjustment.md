# Spot Adjustment Methodology

Option premiums are heavily driven by underlying spot movement. Any claim of predictive power from order flow must therefore control for the directional effect of spot.

## Why Adjustment Is Necessary
A positive relationship between flow and future premium change can arise mechanically if both are responding to the same move in the underlying. Without adjustment, this may falsely appear as flow alpha.

## Residual Premium Construction
For each option file:

1. compute future premium change over the chosen horizon
2. compute the corresponding spot change over the same horizon
3. define directional spot effect according to option type:
   - CE uses positive spot change
   - PE uses negative spot change
4. regress future premium change on the directional spot effect
5. treat the regression residual as the non-spot component of premium movement

The residual is the key quantity used to test whether order flow contains independent predictive information.
