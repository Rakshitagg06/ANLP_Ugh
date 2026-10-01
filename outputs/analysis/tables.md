### Main results

| Model | Params | DET macro-F1 | TYPE macro-F1 | TYPE macro-P | TYPE macro-R | LVR-A | LVR-B | LVR-total |
|---|---|---|---|---|---|---|---|---|
| M1 independent | 368M | 0.799 ± 0.007 | 0.543 ± 0.013 | 0.509 ± 0.024 | 0.600 ± 0.023 | 61.9 ± 1.2% | 0.0 ± 0.0% | 61.9 ± 1.2% |
| M2 shared MTL | 184M | 0.793 ± 0.003 | 0.497 ± 0.015 | 0.444 ± 0.018 | 0.595 ± 0.019 | 61.5 ± 1.5% | 0.0 ± 0.0% | 61.6 ± 1.5% |
| M3 one-way | 184M | 0.793 ± 0.003 | 0.453 ± 0.022 | 0.469 ± 0.033 | 0.468 ± 0.036 | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% |
| M3 symmetric | 184M | 0.793 ± 0.003 | 0.453 ± 0.022 | 0.469 ± 0.033 | 0.468 ± 0.036 | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% |
| M4-core noisy-OR | 184M | 0.784 ± 0.004 | 0.461 ± 0.007 | 0.465 ± 0.011 | 0.474 ± 0.022 | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% |

### Secondary metrics

| Model | DET precision (pos.) | DET recall (pos.) | TYPE macro-F1 (all ex.) | TYPE macro-P (all ex.) | TYPE macro-R (all ex.) | Label recall, multi-label ex. | Label recall, single-label ex. |
|---|---|---|---|---|---|---|---|
| M1 independent | 0.733 ± 0.002 | 0.761 ± 0.026 | 0.299 ± 0.013 | 0.203 ± 0.014 | 0.600 ± 0.023 | 0.743 ± 0.020 | 0.987 ± 0.002 |
| M2 shared MTL | 0.722 ± 0.009 | 0.761 ± 0.021 | 0.255 ± 0.029 | 0.169 ± 0.023 | 0.595 ± 0.019 | 0.746 ± 0.017 | 0.987 ± 0.004 |
| M3 one-way | 0.722 ± 0.009 | 0.761 ± 0.021 | 0.382 ± 0.017 | 0.339 ± 0.019 | 0.468 ± 0.036 | 0.585 ± 0.033 | 0.748 ± 0.027 |
| M3 symmetric | 0.722 ± 0.009 | 0.761 ± 0.021 | 0.382 ± 0.017 | 0.339 ± 0.019 | 0.468 ± 0.036 | 0.585 ± 0.033 | 0.748 ± 0.027 |
| M4-core noisy-OR | 0.674 ± 0.008 | 0.831 ± 0.015 | 0.396 ± 0.008 | 0.348 ± 0.008 | 0.474 ± 0.022 | 0.610 ± 0.018 | 0.805 ± 0.010 |

### Per-label F1 (gold-polarized examples)

| Model | Political (n=1150) | Racial/ethnic (n=281) | Religious (n=112) | Gender/sexual (n=72) | Other (n=126) |
|---|---|---|---|---|---|
| M1 independent | 0.985 ± 0.005 | 0.623 ± 0.024 | 0.548 ± 0.019 | 0.266 ± 0.041 | 0.293 ± 0.019 |
| M2 shared MTL | 0.986 ± 0.003 | 0.583 ± 0.021 | 0.477 ± 0.035 | 0.200 ± 0.060 | 0.238 ± 0.019 |
| M3 one-way | 0.852 ± 0.015 | 0.547 ± 0.020 | 0.456 ± 0.030 | 0.175 ± 0.081 | 0.237 ± 0.022 |
| M3 symmetric | 0.852 ± 0.015 | 0.547 ± 0.020 | 0.456 ± 0.030 | 0.175 ± 0.081 | 0.237 ± 0.022 |
| M4-core noisy-OR | 0.892 ± 0.012 | 0.521 ± 0.013 | 0.450 ± 0.017 | 0.214 ± 0.035 | 0.228 ± 0.007 |

**Per-label precision**

| Model | Political | Racial/ethnic | Religious | Gender/sexual | Other |
|---|---|---|---|---|---|
| M1 independent | 0.979 ± 0.001 | 0.565 ± 0.033 | 0.508 ± 0.041 | 0.265 ± 0.078 | 0.227 ± 0.022 |
| M2 shared MTL | 0.979 ± 0.001 | 0.494 ± 0.034 | 0.409 ± 0.069 | 0.169 ± 0.075 | 0.169 ± 0.023 |
| M3 one-way | 0.979 ± 0.003 | 0.532 ± 0.044 | 0.424 ± 0.067 | 0.235 ± 0.121 | 0.178 ± 0.030 |
| M3 symmetric | 0.979 ± 0.003 | 0.532 ± 0.044 | 0.424 ± 0.067 | 0.235 ± 0.121 | 0.178 ± 0.030 |
| M4-core noisy-OR | 0.978 ± 0.002 | 0.490 ± 0.012 | 0.425 ± 0.022 | 0.241 ± 0.040 | 0.191 ± 0.028 |

**Per-label recall**

| Model | Political | Racial/ethnic | Religious | Gender/sexual | Other |
|---|---|---|---|---|---|
| M1 independent | 0.991 ± 0.009 | 0.695 ± 0.040 | 0.598 ± 0.036 | 0.289 ± 0.051 | 0.424 ± 0.067 |
| M2 shared MTL | 0.992 ± 0.005 | 0.714 ± 0.032 | 0.584 ± 0.023 | 0.269 ± 0.035 | 0.417 ± 0.073 |
| M3 one-way | 0.754 ± 0.022 | 0.569 ± 0.049 | 0.502 ± 0.024 | 0.147 ± 0.069 | 0.370 ± 0.057 |
| M3 symmetric | 0.754 ± 0.022 | 0.569 ± 0.049 | 0.502 ± 0.024 | 0.147 ± 0.069 | 0.370 ± 0.057 |
| M4-core noisy-OR | 0.821 ± 0.020 | 0.557 ± 0.033 | 0.486 ± 0.070 | 0.211 ± 0.073 | 0.297 ± 0.053 |

### Paired comparisons (difference = A − B; 95% bootstrap CI; 2,000 resamples)

| Tier | RQ | A − B | Metric | A | B | Diff | 95% CI | p (boot) | p (Holm) | Seeds A>B |
|---|---|---|---|---|---|---|---|---|---|---|
| primary | RQ1 | M4-core noisy-OR − M1 independent | DET macro-F1 | 0.784 | 0.799 | -0.014 | [-0.023, -0.006] | 0.000 | 0.000 | 0/5 |
| primary | RQ1 | M4-core noisy-OR − M1 independent | TYPE macro-F1 | 0.461 | 0.543 | -0.082 | [-0.102, -0.064] | 0.000 | 0.000 | 0/5 |
| primary | RQ1 | M4-core noisy-OR − M2 shared MTL | DET macro-F1 | 0.784 | 0.793 | -0.008 | [-0.016, -0.001] | 0.024 | 0.072 | 0/5 |
| primary | RQ1 | M4-core noisy-OR − M2 shared MTL | TYPE macro-F1 | 0.461 | 0.497 | -0.036 | [-0.055, -0.019] | 0.001 | 0.004 | 0/5 |
| primary | RQ2 | M4-core noisy-OR − M3 symmetric | TYPE macro-F1 | 0.461 | 0.453 | +0.008 | [-0.009, +0.022] | 0.366 | 0.732 | 3/5 |
| primary | RQ2 | M4-core noisy-OR − M3 symmetric | TYPE macro-R | 0.474 | 0.468 | +0.006 | [-0.014, +0.023] | 0.561 | 0.732 | 2/5 |
| secondary | RQ1 | M4-core noisy-OR − M1 independent | TYPE macro-P | 0.465 | 0.509 | -0.044 | [-0.064, -0.026] | 0.000 |  | 0/5 |
| secondary | RQ1 | M4-core noisy-OR − M1 independent | TYPE macro-R | 0.474 | 0.600 | -0.125 | [-0.151, -0.101] | 0.000 |  | 0/5 |
| secondary | RQ1 | M4-core noisy-OR − M2 shared MTL | TYPE macro-P | 0.465 | 0.444 | +0.021 | [+0.001, +0.039] | 0.037 |  | 4/5 |
| secondary | RQ1 | M4-core noisy-OR − M2 shared MTL | TYPE macro-R | 0.474 | 0.595 | -0.121 | [-0.148, -0.098] | 0.000 |  | 0/5 |
| secondary | RQ1 | M4-core noisy-OR − M1 independent | DET precision (pos.) | 0.674 | 0.733 | -0.059 | [-0.069, -0.048] | 0.000 |  | 0/5 |
| secondary | RQ1 | M4-core noisy-OR − M1 independent | DET recall (pos.) | 0.831 | 0.761 | +0.069 | [+0.057, +0.082] | 0.000 |  | 5/5 |
| secondary | RQ1 | M4-core noisy-OR − M2 shared MTL | DET precision (pos.) | 0.674 | 0.722 | -0.048 | [-0.057, -0.038] | 0.000 |  | 0/5 |
| secondary | RQ1 | M4-core noisy-OR − M2 shared MTL | DET recall (pos.) | 0.831 | 0.761 | +0.070 | [+0.059, +0.081] | 0.000 |  | 5/5 |
| secondary | RQ1 | M4-core noisy-OR − M1 independent | TYPE macro-F1 (all ex.) | 0.396 | 0.299 | +0.097 | [+0.078, +0.114] | 0.000 |  | 5/5 |
| secondary | RQ1 | M4-core noisy-OR − M2 shared MTL | TYPE macro-F1 (all ex.) | 0.396 | 0.255 | +0.141 | [+0.121, +0.158] | 0.000 |  | 5/5 |
| secondary | RQ1 | M2 shared MTL − M1 independent | DET macro-F1 | 0.793 | 0.799 | -0.006 | [-0.011, -0.000] | 0.039 |  | 1/5 |
| secondary | RQ1 | M2 shared MTL − M1 independent | TYPE macro-F1 | 0.497 | 0.543 | -0.046 | [-0.060, -0.033] | 0.000 |  | 0/5 |
| secondary | RQ2 | M4-core noisy-OR − M3 symmetric | TYPE macro-P | 0.465 | 0.469 | -0.004 | [-0.022, +0.012] | 0.553 |  | 2/5 |
| secondary | RQ2 | M4-core noisy-OR − M3 symmetric | DET macro-F1 | 0.784 | 0.793 | -0.008 | [-0.016, -0.001] | 0.026 |  | 0/5 |
| secondary | RQ2 | M4-core noisy-OR − M3 symmetric | Label recall, multi-label ex. | 0.610 | 0.585 | +0.025 | [+0.009, +0.041] | 0.001 |  | 4/5 |
| secondary | RQ2 | M4-core noisy-OR − M3 symmetric | TYPE macro-F1 (all ex.) | 0.396 | 0.382 | +0.014 | [-0.001, +0.026] | 0.073 |  | 4/5 |
| secondary | RQ2 | M3 symmetric − M2 shared MTL | TYPE macro-F1 | 0.453 | 0.497 | -0.043 | [-0.056, -0.032] | 0.000 |  | 0/5 |
| secondary | RQ2 | M3 symmetric − M2 shared MTL | TYPE macro-R | 0.468 | 0.595 | -0.127 | [-0.145, -0.110] | 0.000 |  | 0/5 |
| secondary | RQ2 | M3 symmetric − M2 shared MTL | TYPE macro-P | 0.469 | 0.444 | +0.025 | [+0.013, +0.038] | 0.000 |  | 5/5 |
| secondary | RQ2 | M3 symmetric − M2 shared MTL | DET macro-F1 | 0.793 | 0.793 | -0.000 | [-0.000, +0.000] | 0.242 |  | 0/5 |
| secondary | RQ2 | M3 symmetric − M2 shared MTL | TYPE macro-F1 (all ex.) | 0.382 | 0.255 | +0.127 | [+0.113, +0.142] | 0.000 |  | 5/5 |
| secondary | RQ2 | M3 symmetric − M2 shared MTL | Label recall, multi-label ex. | 0.585 | 0.746 | -0.161 | [-0.184, -0.137] | 0.000 |  | 0/5 |

### Per-label paired differences (A − B)

| A − B | Label | Stat | Diff | 95% CI | p (boot) |
|---|---|---|---|---|---|
| M4-core noisy-OR − M3 symmetric | Political | recall | +0.067 | [+0.055, +0.078] | 0.000 |
| M4-core noisy-OR − M3 symmetric | Political | precision | -0.001 | [-0.004, +0.001] | 0.308 |
| M4-core noisy-OR − M3 symmetric | Political | f1 | +0.040 | [+0.033, +0.048] | 0.000 |
| M4-core noisy-OR − M3 symmetric | Racial/ethnic | recall | -0.011 | [-0.041, +0.019] | 0.430 |
| M4-core noisy-OR − M3 symmetric | Racial/ethnic | precision | -0.042 | [-0.068, -0.016] | 0.000 |
| M4-core noisy-OR − M3 symmetric | Racial/ethnic | f1 | -0.026 | [-0.051, -0.003] | 0.028 |
| M4-core noisy-OR − M3 symmetric | Religious | recall | -0.016 | [-0.059, +0.026] | 0.503 |
| M4-core noisy-OR − M3 symmetric | Religious | precision | +0.001 | [-0.035, +0.039] | 0.956 |
| M4-core noisy-OR − M3 symmetric | Religious | f1 | -0.006 | [-0.041, +0.030] | 0.740 |
| M4-core noisy-OR − M3 symmetric | Gender/sexual | recall | +0.064 | [+0.009, +0.116] | 0.024 |
| M4-core noisy-OR − M3 symmetric | Gender/sexual | precision | +0.006 | [-0.061, +0.067] | 0.926 |
| M4-core noisy-OR − M3 symmetric | Gender/sexual | f1 | +0.039 | [-0.014, +0.089] | 0.165 |
| M4-core noisy-OR − M3 symmetric | Other | recall | -0.073 | [-0.113, -0.034] | 0.000 |
| M4-core noisy-OR − M3 symmetric | Other | precision | +0.013 | [-0.013, +0.040] | 0.320 |
| M4-core noisy-OR − M3 symmetric | Other | f1 | -0.009 | [-0.039, +0.019] | 0.541 |
| M4-core noisy-OR − M2 shared MTL | Political | recall | -0.171 | [-0.189, -0.153] | 0.000 |
| M4-core noisy-OR − M2 shared MTL | Political | precision | -0.001 | [-0.004, +0.001] | 0.314 |
| M4-core noisy-OR − M2 shared MTL | Political | f1 | -0.093 | [-0.104, -0.083] | 0.000 |
| M4-core noisy-OR − M2 shared MTL | Racial/ethnic | recall | -0.157 | [-0.199, -0.116] | 0.000 |
| M4-core noisy-OR − M2 shared MTL | Racial/ethnic | precision | -0.004 | [-0.036, +0.024] | 0.780 |
| M4-core noisy-OR − M2 shared MTL | Racial/ethnic | f1 | -0.062 | [-0.096, -0.032] | 0.000 |
| M4-core noisy-OR − M2 shared MTL | Religious | recall | -0.098 | [-0.154, -0.045] | 0.001 |
| M4-core noisy-OR − M2 shared MTL | Religious | precision | +0.016 | [-0.024, +0.057] | 0.435 |
| M4-core noisy-OR − M2 shared MTL | Religious | f1 | -0.027 | [-0.070, +0.015] | 0.228 |
| M4-core noisy-OR − M2 shared MTL | Gender/sexual | recall | -0.058 | [-0.140, +0.012] | 0.122 |
| M4-core noisy-OR − M2 shared MTL | Gender/sexual | precision | +0.072 | [-0.001, +0.142] | 0.056 |
| M4-core noisy-OR − M2 shared MTL | Gender/sexual | f1 | +0.014 | [-0.054, +0.073] | 0.750 |
| M4-core noisy-OR − M2 shared MTL | Other | recall | -0.121 | [-0.165, -0.077] | 0.000 |
| M4-core noisy-OR − M2 shared MTL | Other | precision | +0.022 | [-0.006, +0.051] | 0.128 |
| M4-core noisy-OR − M2 shared MTL | Other | f1 | -0.010 | [-0.043, +0.021] | 0.549 |
| M4-core noisy-OR − M1 independent | Political | recall | -0.171 | [-0.188, -0.153] | 0.000 |
| M4-core noisy-OR − M1 independent | Political | precision | -0.001 | [-0.004, +0.001] | 0.279 |
| M4-core noisy-OR − M1 independent | Political | f1 | -0.093 | [-0.104, -0.082] | 0.000 |
| M4-core noisy-OR − M1 independent | Racial/ethnic | recall | -0.138 | [-0.181, -0.095] | 0.000 |
| M4-core noisy-OR − M1 independent | Racial/ethnic | precision | -0.075 | [-0.109, -0.043] | 0.000 |
| M4-core noisy-OR − M1 independent | Racial/ethnic | f1 | -0.102 | [-0.137, -0.071] | 0.000 |
| M4-core noisy-OR − M1 independent | Religious | recall | -0.112 | [-0.173, -0.055] | 0.000 |
| M4-core noisy-OR − M1 independent | Religious | precision | -0.083 | [-0.131, -0.039] | 0.000 |
| M4-core noisy-OR − M1 independent | Religious | f1 | -0.098 | [-0.143, -0.053] | 0.000 |
| M4-core noisy-OR − M1 independent | Gender/sexual | recall | -0.078 | [-0.144, -0.021] | 0.008 |
| M4-core noisy-OR − M1 independent | Gender/sexual | precision | -0.024 | [-0.094, +0.036] | 0.402 |
| M4-core noisy-OR − M1 independent | Gender/sexual | f1 | -0.052 | [-0.114, -0.001] | 0.047 |
| M4-core noisy-OR − M1 independent | Other | recall | -0.127 | [-0.184, -0.071] | 0.000 |
| M4-core noisy-OR − M1 independent | Other | precision | -0.036 | [-0.070, -0.004] | 0.031 |
| M4-core noisy-OR − M1 independent | Other | f1 | -0.065 | [-0.104, -0.026] | 0.002 |

### M2 → M3 gating audit (per seed over all 3,222 examples; mean ± sd)

| Quantity | Value |
|---|---|
| m2 type labels predicted | 6647.0 ± 774.9 |
| labels removed | 4379.4 ± 766.9 |
| removed on gold negative | 3838.6 ± 726.4 |
| removed fp on gold positive | 202.4 ± 51.4 |
| removed tp on gold positive | 338.4 ± 29.2 |
| examples gated | 1983.0 ± 48.9 |
| gold positive examples gated | 280.8 ± 24.6 |
| gold positive examples gated with correct type | 277.2 ± 26.4 |
| det predictions changed | 0.4 ± 0.9 |
| m2 det false negatives | 280.8 ± 24.6 |
| m2 lvr b examples | 0.4 ± 0.9 |

### DET error profile (mean per seed over 3,222 examples)

| Model | false positives | false negatives | predicted positive | type labels on gold negative | gold negative with any type |
|---|---|---|---|---|---|
| M1 independent | 325.4 | 280.4 | 1220.0 | 3296.6 | 2039.2 |
| M2 shared MTL | 344.8 | 280.8 | 1239.0 | 4455.6 | 2047.0 |
| M3 one-way | 344.8 | 280.8 | 1239.0 | 617.0 | 344.8 |
| M3 symmetric | 344.8 | 281.2 | 1238.6 | 617.0 | 344.8 |
| M4-core noisy-OR | 471.6 | 199.0 | 1447.6 | 642.8 | 471.6 |

### Training statistics (25 runs per trained condition)

| Run type | best epoch mean | best epoch min | best epoch max | best epoch at cap | epochs run mean | runtime min mean | runtime min total |
|---|---|---|---|---|---|---|---|
| m1_det | 2.8 | 1.0 | 5.0 | 0.0 | 4.8 | 1.9 | 46.3 |
| m1_type | 4.1 | 3.0 | 6.0 | 0.0 | 6.1 | 2.4 | 59.3 |
| m2 | 3.9 | 2.0 | 8.0 | 2.0 | 5.8 | 2.3 | 56.6 |
| m4_core | 5.8 | 4.0 | 8.0 | 1.0 | 7.5 | 2.9 | 73.2 |
