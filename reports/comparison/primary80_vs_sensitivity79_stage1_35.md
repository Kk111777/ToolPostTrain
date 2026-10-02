# Primary 80 vs sensitivity 79

The primary endpoint is the frozen 80-row formal validation. The sensitivity endpoint removes the pre-registered source_id 3814 row from exactly those persisted outputs; it does not resample or redefine validation.

| algorithm | step | primary total | sensitivity total | delta sensitivity-primary | primary accuracy | sensitivity accuracy |
|---|---:|---:|---:|---:|---:|---:|
| GRPO-original | 0 | 2.400171 | 2.405237 | 0.005065 | 1.450171 | 1.455869 |
| GRPO-original | 7 | 2.717582 | 2.726665 | 0.009083 | 1.767582 | 1.777298 |
| GRPO-original | 14 | 2.799781 | 2.809904 | 0.010124 | 1.849781 | 1.860537 |
| GRPO-original | 21 | 2.785379 | 2.795321 | 0.009942 | 1.835379 | 1.845954 |
| GRPO-original | 28 | 2.831808 | 2.842337 | 0.010529 | 1.881808 | 1.892970 |
| GRPO-original | 35 | 2.799721 | 2.809844 | 0.010123 | 1.849721 | 1.860477 |
| GDPO-current | 0 | 2.400171 | 2.405237 | 0.005065 | 1.450171 | 1.455869 |
| GDPO-current | 7 | 2.611385 | 2.619124 | 0.007739 | 1.648885 | 1.657099 |
| GDPO-current | 14 | 2.798260 | 2.808365 | 0.010105 | 1.835760 | 1.846340 |
| GDPO-current | 21 | 2.768737 | 2.778467 | 0.009731 | 1.806237 | 1.816442 |
| GDPO-current | 28 | 2.781237 | 2.791126 | 0.009889 | 1.818737 | 1.829100 |
| GDPO-current | 35 | 2.820522 | 2.830909 | 0.010386 | 1.858022 | 1.868883 |
| GRPO-noKL | 0 | 2.400171 | 2.405237 | 0.005065 | 1.450171 | 1.455869 |
| GRPO-noKL | 7 | 2.588245 | 2.595691 | 0.007446 | 1.638245 | 1.646324 |
| GRPO-noKL | 14 | 2.757760 | 2.767352 | 0.009592 | 1.807760 | 1.817985 |
| GRPO-noKL | 21 | 2.784451 | 2.794381 | 0.009930 | 1.821951 | 1.832355 |
| GRPO-noKL | 28 | 2.781237 | 2.791126 | 0.009889 | 1.818737 | 1.829100 |
| GRPO-noKL | 35 | 2.818737 | 2.829100 | 0.010364 | 1.856237 | 1.867075 |

The excluded row is held fixed across algorithms and steps. Interpret alongside the KL-treatment/reward-dimension confounds and the single-seed, 35-step budget.
