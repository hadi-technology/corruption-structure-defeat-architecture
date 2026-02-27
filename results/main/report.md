# Experiment Report

## Hardware
- Platform: `Linux-6.17.0-14-generic-x86_64-with-glibc2.39`
- CPU logical cores: `8`
- RAM total: `31.24 GiB`
- RAM available at run start: `13.9 GiB`
- Disk free: `12.13 GiB`
- GPU: not available to `nvidia-smi` in this run context

## RuleTaker Main Results
### Clean
| Model | Accuracy (mean±std) | Cascade (mean±std) | Chain-Linked Error (mean±std) | V2 Transient Use (mean±std) |
|---|---:|---:|---:|---:|
| instance_weighted | 0.928 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | - |
| aspic_v1_terminal | 0.928 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | - |
| aspic_v2 | 0.928 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| aspic_v2_persistent | 0.928 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| schema_gating | 0.928 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | - |

### Random (Negate vs Redirect)
| Model | Negate Acc | Negate Cascade | Negate Chain | Redirect Acc | Redirect Cascade | Redirect Chain |
|---|---:|---:|---:|---:|---:|---:|
| instance_weighted | 0.775 | 0.003 | 0.259 | 0.782 | 0.065 | 0.143 |
| aspic_v1_terminal | 0.775 | 0.000 | 0.257 | 0.764 | 0.000 | 0.073 |
| aspic_v2 | 0.769 | 0.000 | 0.193 | 0.780 | 0.058 | 0.091 |
| aspic_v2_persistent | 0.770 | 0.000 | 0.194 | 0.785 | 0.062 | 0.096 |
| schema_gating | 0.765 | 0.000 | 0.187 | 0.777 | 0.045 | 0.078 |

### Clustered (Negate vs Redirect)
| Model | Negate Acc | Negate Cascade | Negate Chain | Redirect Acc | Redirect Cascade | Redirect Chain |
|---|---:|---:|---:|---:|---:|---:|
| instance_weighted | 0.627 | 0.002 | 0.175 | 0.669 | 0.066 | 0.137 |
| aspic_v1_terminal | 0.627 | 0.000 | 0.174 | 0.645 | 0.000 | 0.064 |
| aspic_v2 | 0.601 | 0.000 | 0.044 | 0.623 | 0.018 | 0.046 |
| aspic_v2_persistent | 0.607 | 0.000 | 0.046 | 0.637 | 0.037 | 0.067 |
| schema_gating | 0.599 | 0.000 | 0.043 | 0.609 | 0.003 | 0.029 |


- `Cascade` is defined as `corrupted_chain / incorrect` (paper definition).
- `Chain-Linked Error` is `incorrect_with_chain / incorrect` (diagnostic coverage metric).
- `aspic_v2_persistent` is the report label for `aspic_v2_window1`.
- It keeps blocked transient nodes visible for one extra step; under redirect corruption this can approach instance-weighted behavior via cross-step consumption.

## RuleTaker Mechanism Checks
- Clean false-negative rate (reliable schemas removed): `0.00%`
- Clustered schema construction reduction (gating): `46.37%`
- Low-confidence fallback decision rates (policy=`closed_world`):
  - `clean/instance_weighted`: `60.00%`
  - `clean/aspic_v1_terminal`: `60.00%`
  - `clean/aspic_v2`: `60.00%`
  - `clean/aspic_v2_persistent`: `60.00%`
  - `clean/schema_gating`: `60.00%`
  - `random_negate/instance_weighted`: `70.87%`
  - `random_negate/aspic_v1_terminal`: `70.93%`
  - `random_negate/aspic_v2`: `71.67%`
  - `random_negate/aspic_v2_persistent`: `71.60%`
  - `random_negate/schema_gating`: `71.93%`
  - `clustered_negate/instance_weighted`: `85.60%`
  - `clustered_negate/aspic_v1_terminal`: `85.67%`
  - `clustered_negate/aspic_v2`: `91.20%`
  - `clustered_negate/aspic_v2_persistent`: `90.60%`
  - `clustered_negate/schema_gating`: `91.40%`
  - `random_redirect/instance_weighted`: `70.67%`
  - `random_redirect/aspic_v1_terminal`: `74.27%`
  - `random_redirect/aspic_v2`: `71.93%`
  - `random_redirect/aspic_v2_persistent`: `71.27%`
  - `random_redirect/schema_gating`: `71.60%`
  - `clustered_redirect/instance_weighted`: `82.73%`
  - `clustered_redirect/aspic_v1_terminal`: `86.40%`
  - `clustered_redirect/aspic_v2`: `88.73%`
  - `clustered_redirect/aspic_v2_persistent`: `86.87%`
  - `clustered_redirect/schema_gating`: `90.00%`
- ASPIC+ transient propagation window usage:
  - `clean/aspic_v2`: blocked `0`, consumed `0`, utilization `0.00%`
  - `clean/aspic_v2_persistent`: blocked `0`, consumed `0`, utilization `0.00%`
  - `random_negate/aspic_v2`: blocked `1625`, consumed `186`, utilization `11.45%`
  - `random_negate/aspic_v2_persistent`: blocked `4921`, consumed `691`, utilization `14.04%`
  - `clustered_negate/aspic_v2`: blocked `5243`, consumed `314`, utilization `5.99%`
  - `clustered_negate/aspic_v2_persistent`: blocked `21396`, consumed `1987`, utilization `9.29%`
  - `random_redirect/aspic_v2`: blocked `1857`, consumed `775`, utilization `41.73%`
  - `random_redirect/aspic_v2_persistent`: blocked `5768`, consumed `3408`, utilization `59.08%`
  - `clustered_redirect/aspic_v2`: blocked `6997`, consumed `1804`, utilization `25.78%`
  - `clustered_redirect/aspic_v2_persistent`: blocked `29931`, consumed `10609`, utilization `35.44%`

## RuleTaker Error Decomposition
| Condition | Model | CWA Error | Chain-Linked Error | Direct Error |
|---|---|---:|---:|---:|
| clean | instance_weighted | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| clean | aspic_v1_terminal | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| clean | aspic_v2 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| clean | aspic_v2_persistent | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| clean | schema_gating | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| random_negate | instance_weighted | 0.793 ± 0.005 | 0.130 ± 0.003 | 0.077 ± 0.007 |
| random_negate | aspic_v1_terminal | 0.793 ± 0.005 | 0.131 ± 0.003 | 0.077 ± 0.007 |
| random_negate | aspic_v2 | 0.807 ± 0.013 | 0.118 ± 0.008 | 0.075 ± 0.008 |
| random_negate | aspic_v2_persistent | 0.806 ± 0.014 | 0.119 ± 0.008 | 0.075 ± 0.008 |
| random_negate | schema_gating | 0.813 ± 0.012 | 0.113 ± 0.009 | 0.073 ± 0.008 |
| clustered_negate | instance_weighted | 0.869 ± 0.021 | 0.056 ± 0.017 | 0.075 ± 0.013 |
| clustered_negate | aspic_v1_terminal | 0.871 ± 0.024 | 0.054 ± 0.018 | 0.075 ± 0.013 |
| clustered_negate | aspic_v2 | 0.956 ± 0.003 | 0.020 ± 0.001 | 0.023 ± 0.003 |
| clustered_negate | aspic_v2_persistent | 0.954 ± 0.004 | 0.022 ± 0.002 | 0.024 ± 0.003 |
| clustered_negate | schema_gating | 0.957 ± 0.003 | 0.020 ± 0.001 | 0.023 ± 0.003 |
| random_redirect | instance_weighted | 0.911 ± 0.019 | 0.077 ± 0.018 | 0.012 ± 0.004 |
| random_redirect | aspic_v1_terminal | 0.963 ± 0.011 | 0.026 ± 0.007 | 0.011 ± 0.004 |
| random_redirect | aspic_v2 | 0.915 ± 0.015 | 0.073 ± 0.013 | 0.012 ± 0.004 |
| random_redirect | aspic_v2_persistent | 0.913 ± 0.016 | 0.074 ± 0.013 | 0.012 ± 0.004 |
| random_redirect | schema_gating | 0.922 ± 0.013 | 0.066 ± 0.009 | 0.012 ± 0.004 |
| clustered_redirect | instance_weighted | 0.925 ± 0.012 | 0.036 ± 0.010 | 0.038 ± 0.002 |
| clustered_redirect | aspic_v1_terminal | 0.964 ± 0.003 | 0.000 ± 0.000 | 0.036 ± 0.003 |
| clustered_redirect | aspic_v2 | 0.963 ± 0.013 | 0.011 ± 0.009 | 0.027 ± 0.005 |
| clustered_redirect | aspic_v2_persistent | 0.948 ± 0.017 | 0.024 ± 0.011 | 0.028 ± 0.005 |
| clustered_redirect | schema_gating | 0.971 ± 0.006 | 0.003 ± 0.002 | 0.026 ± 0.005 |

## RuleTaker Derivable-Only Metrics
| Condition | Model | Derivable Coverage | Derivable Accuracy | Derivable Chain-Linked Error |
|---|---|---:|---:|---:|
| clean | instance_weighted | 0.400 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| clean | aspic_v1_terminal | 0.400 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| clean | aspic_v2 | 0.400 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| clean | aspic_v2_persistent | 0.400 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| clean | schema_gating | 0.400 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| random_negate | instance_weighted | 0.291 | 0.839 ± 0.016 | 0.631 ± 0.027 |
| random_negate | aspic_v1_terminal | 0.291 | 0.839 ± 0.016 | 0.631 ± 0.027 |
| random_negate | aspic_v2 | 0.283 | 0.842 ± 0.019 | 0.613 ± 0.021 |
| random_negate | aspic_v2_persistent | 0.284 | 0.842 ± 0.019 | 0.613 ± 0.021 |
| random_negate | schema_gating | 0.281 | 0.843 ± 0.019 | 0.607 ± 0.030 |
| clustered_negate | instance_weighted | 0.144 | 0.660 ± 0.058 | 0.424 ± 0.076 |
| clustered_negate | aspic_v1_terminal | 0.143 | 0.664 ± 0.062 | 0.414 ± 0.078 |
| clustered_negate | aspic_v2 | 0.088 | 0.799 ± 0.030 | 0.463 ± 0.026 |
| clustered_negate | aspic_v2_persistent | 0.094 | 0.805 ± 0.035 | 0.481 ± 0.026 |
| clustered_negate | schema_gating | 0.086 | 0.795 ± 0.028 | 0.463 ± 0.026 |
| random_redirect | instance_weighted | 0.293 | 0.934 ± 0.011 | 0.861 ± 0.037 |
| random_redirect | aspic_v1_terminal | 0.257 | 0.967 ± 0.008 | 0.694 ± 0.039 |
| random_redirect | aspic_v2 | 0.281 | 0.934 ± 0.012 | 0.858 ± 0.033 |
| random_redirect | aspic_v2_persistent | 0.287 | 0.935 ± 0.011 | 0.858 ± 0.033 |
| random_redirect | schema_gating | 0.284 | 0.939 ± 0.012 | 0.850 ± 0.024 |
| clustered_redirect | instance_weighted | 0.173 | 0.858 ± 0.011 | 0.478 ± 0.057 |
| clustered_redirect | aspic_v1_terminal | 0.136 | 0.907 ± 0.004 | 0.000 ± 0.000 |
| clustered_redirect | aspic_v2 | 0.113 | 0.878 ± 0.018 | 0.244 ± 0.175 |
| clustered_redirect | aspic_v2_persistent | 0.131 | 0.860 ± 0.014 | 0.447 ± 0.068 |
| clustered_redirect | schema_gating | 0.100 | 0.887 ± 0.005 | 0.114 ± 0.084 |

## ProofWriter Main Results
### Clean
| Model | Accuracy (mean±std) | Cascade (mean±std) | Chain-Linked Error (mean±std) | V2 Transient Use (mean±std) |
|---|---:|---:|---:|---:|
| instance_weighted | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | - |
| aspic_v1_terminal | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | - |
| aspic_v2 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| aspic_v2_persistent | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| schema_gating | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | - |

### Random (Negate vs Redirect)
| Model | Negate Acc | Negate Cascade | Negate Chain | Redirect Acc | Redirect Cascade | Redirect Chain |
|---|---:|---:|---:|---:|---:|---:|
| instance_weighted | 0.756 | 0.000 | 0.490 | 0.852 | 0.008 | 0.015 |
| aspic_v1_terminal | 0.754 | 0.000 | 0.487 | 0.816 | 0.000 | 0.006 |
| aspic_v2 | 0.756 | 0.000 | 0.399 | 0.846 | 0.007 | 0.015 |
| aspic_v2_persistent | 0.760 | 0.000 | 0.407 | 0.849 | 0.008 | 0.015 |
| schema_gating | 0.753 | 0.000 | 0.387 | 0.843 | 0.008 | 0.015 |

### Clustered (Negate vs Redirect)
| Model | Negate Acc | Negate Cascade | Negate Chain | Redirect Acc | Redirect Cascade | Redirect Chain |
|---|---:|---:|---:|---:|---:|---:|
| instance_weighted | 0.557 | 0.000 | 0.436 | 0.673 | 0.003 | 0.007 |
| aspic_v1_terminal | 0.557 | 0.000 | 0.436 | 0.644 | 0.000 | 0.003 |
| aspic_v2 | 0.576 | 0.000 | 0.148 | 0.619 | 0.000 | 0.003 |
| aspic_v2_persistent | 0.579 | 0.000 | 0.157 | 0.640 | 0.003 | 0.006 |
| schema_gating | 0.574 | 0.000 | 0.139 | 0.619 | 0.000 | 0.003 |


- `Cascade` is defined as `corrupted_chain / incorrect` (paper definition).
- `Chain-Linked Error` is `incorrect_with_chain / incorrect` (diagnostic coverage metric).
- `aspic_v2_persistent` is the report label for `aspic_v2_window1`.
- It keeps blocked transient nodes visible for one extra step; under redirect corruption this can approach instance-weighted behavior via cross-step consumption.

## ProofWriter Mechanism Checks
- Clean false-negative rate (reliable schemas removed): `0.00%`
- Clustered schema construction reduction (gating): `42.34%`
- Low-confidence fallback decision rates (policy=`random_tie`):
  - `clean/instance_weighted`: `0.00%`
  - `clean/aspic_v1_terminal`: `0.00%`
  - `clean/aspic_v2`: `0.00%`
  - `clean/aspic_v2_persistent`: `0.00%`
  - `clean/schema_gating`: `0.00%`
  - `random_negate/instance_weighted`: `29.33%`
  - `random_negate/aspic_v1_terminal`: `29.44%`
  - `random_negate/aspic_v2`: `31.33%`
  - `random_negate/aspic_v2_persistent`: `30.67%`
  - `random_negate/schema_gating`: `32.11%`
  - `clustered_negate/instance_weighted`: `63.67%`
  - `clustered_negate/aspic_v1_terminal`: `63.67%`
  - `clustered_negate/aspic_v2`: `74.67%`
  - `clustered_negate/aspic_v2_persistent`: `73.56%`
  - `clustered_negate/schema_gating`: `76.11%`
  - `random_redirect/instance_weighted`: `31.56%`
  - `random_redirect/aspic_v1_terminal`: `37.67%`
  - `random_redirect/aspic_v2`: `33.33%`
  - `random_redirect/aspic_v2_persistent`: `32.56%`
  - `random_redirect/schema_gating`: `33.56%`
  - `clustered_redirect/instance_weighted`: `66.78%`
  - `clustered_redirect/aspic_v1_terminal`: `73.22%`
  - `clustered_redirect/aspic_v2`: `78.89%`
  - `clustered_redirect/aspic_v2_persistent`: `74.67%`
  - `clustered_redirect/schema_gating`: `79.22%`
- ASPIC+ transient propagation window usage:
  - `clean/aspic_v2`: blocked `0`, consumed `0`, utilization `0.00%`
  - `clean/aspic_v2_persistent`: blocked `0`, consumed `0`, utilization `0.00%`
  - `random_negate/aspic_v2`: blocked `1145`, consumed `183`, utilization `15.98%`
  - `random_negate/aspic_v2_persistent`: blocked `3216`, consumed `650`, utilization `20.21%`
  - `clustered_negate/aspic_v2`: blocked `4478`, consumed `267`, utilization `5.96%`
  - `clustered_negate/aspic_v2_persistent`: blocked `16916`, consumed `1564`, utilization `9.25%`
  - `random_redirect/aspic_v2`: blocked `1715`, consumed `868`, utilization `50.61%`
  - `random_redirect/aspic_v2_persistent`: blocked `4495`, consumed `2948`, utilization `65.58%`
  - `clustered_redirect/aspic_v2`: blocked `6449`, consumed `1843`, utilization `28.58%`
  - `clustered_redirect/aspic_v2_persistent`: blocked `24953`, consumed `9857`, utilization `39.50%`

## ProofWriter Error Decomposition
| Condition | Model | CWA Error | Chain-Linked Error | Direct Error |
|---|---|---:|---:|---:|
| clean | instance_weighted | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| clean | aspic_v1_terminal | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| clean | aspic_v2 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| clean | aspic_v2_persistent | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| clean | schema_gating | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| random_negate | instance_weighted | 0.565 ± 0.070 | 0.286 ± 0.065 | 0.150 ± 0.016 |
| random_negate | aspic_v1_terminal | 0.567 ± 0.066 | 0.284 ± 0.063 | 0.149 ± 0.016 |
| random_negate | aspic_v2 | 0.601 ± 0.068 | 0.263 ± 0.062 | 0.136 ± 0.008 |
| random_negate | aspic_v2_persistent | 0.593 ± 0.072 | 0.268 ± 0.064 | 0.139 ± 0.009 |
| random_negate | schema_gating | 0.613 ± 0.077 | 0.252 ± 0.069 | 0.135 ± 0.008 |
| clustered_negate | instance_weighted | 0.690 ± 0.013 | 0.133 ± 0.012 | 0.178 ± 0.006 |
| clustered_negate | aspic_v1_terminal | 0.690 ± 0.013 | 0.133 ± 0.012 | 0.178 ± 0.006 |
| clustered_negate | aspic_v2 | 0.852 ± 0.030 | 0.054 ± 0.019 | 0.094 ± 0.011 |
| clustered_negate | aspic_v2_persistent | 0.843 ± 0.024 | 0.063 ± 0.013 | 0.094 ± 0.012 |
| clustered_negate | schema_gating | 0.861 ± 0.032 | 0.046 ± 0.020 | 0.093 ± 0.012 |
| random_redirect | instance_weighted | 0.985 ± 0.011 | 0.008 ± 0.011 | 0.007 ± 0.010 |
| random_redirect | aspic_v1_terminal | 0.994 ± 0.008 | 0.000 ± 0.000 | 0.006 ± 0.008 |
| random_redirect | aspic_v2 | 0.985 ± 0.010 | 0.007 ± 0.010 | 0.007 ± 0.010 |
| random_redirect | aspic_v2_persistent | 0.985 ± 0.011 | 0.008 ± 0.011 | 0.007 ± 0.010 |
| random_redirect | schema_gating | 0.985 ± 0.011 | 0.008 ± 0.011 | 0.007 ± 0.010 |
| clustered_redirect | instance_weighted | 0.997 ± 0.005 | 0.000 ± 0.000 | 0.003 ± 0.005 |
| clustered_redirect | aspic_v1_terminal | 0.997 ± 0.005 | 0.000 ± 0.000 | 0.003 ± 0.005 |
| clustered_redirect | aspic_v2 | 0.997 ± 0.004 | 0.000 ± 0.000 | 0.003 ± 0.004 |
| clustered_redirect | aspic_v2_persistent | 0.997 ± 0.004 | 0.000 ± 0.000 | 0.003 ± 0.004 |
| clustered_redirect | schema_gating | 0.997 ± 0.004 | 0.000 ± 0.000 | 0.003 ± 0.004 |

## ProofWriter Derivable-Only Metrics
| Condition | Model | Derivable Coverage | Derivable Accuracy | Derivable Chain-Linked Error |
|---|---|---:|---:|---:|
| clean | instance_weighted | 1.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| clean | aspic_v1_terminal | 1.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| clean | aspic_v2 | 1.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| clean | aspic_v2_persistent | 1.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| clean | schema_gating | 1.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| random_negate | instance_weighted | 0.707 | 0.849 ± 0.027 | 0.650 ± 0.056 |
| random_negate | aspic_v1_terminal | 0.706 | 0.849 ± 0.027 | 0.650 ± 0.056 |
| random_negate | aspic_v2 | 0.687 | 0.857 ± 0.028 | 0.652 ± 0.042 |
| random_negate | aspic_v2_persistent | 0.693 | 0.859 ± 0.027 | 0.652 ± 0.042 |
| random_negate | schema_gating | 0.679 | 0.859 ± 0.030 | 0.642 ± 0.049 |
| clustered_negate | instance_weighted | 0.363 | 0.621 ± 0.033 | 0.427 ± 0.023 |
| clustered_negate | aspic_v1_terminal | 0.363 | 0.621 ± 0.033 | 0.427 ± 0.023 |
| clustered_negate | aspic_v2 | 0.253 | 0.750 ± 0.066 | 0.356 ± 0.048 |
| clustered_negate | aspic_v2_persistent | 0.264 | 0.747 ± 0.060 | 0.396 ± 0.031 |
| clustered_negate | schema_gating | 0.239 | 0.749 ± 0.071 | 0.314 ± 0.069 |
| random_redirect | instance_weighted | 0.684 | 0.997 ± 0.002 | 0.333 ± 0.471 |
| random_redirect | aspic_v1_terminal | 0.623 | 0.998 ± 0.003 | 0.000 ± 0.000 |
| random_redirect | aspic_v2 | 0.667 | 0.997 ± 0.002 | 0.333 ± 0.471 |
| random_redirect | aspic_v2_persistent | 0.674 | 0.997 ± 0.002 | 0.333 ± 0.471 |
| random_redirect | schema_gating | 0.664 | 0.997 ± 0.002 | 0.333 ± 0.471 |
| clustered_redirect | instance_weighted | 0.332 | 0.997 ± 0.005 | 0.000 ± 0.000 |
| clustered_redirect | aspic_v1_terminal | 0.268 | 0.996 ± 0.006 | 0.000 ± 0.000 |
| clustered_redirect | aspic_v2 | 0.211 | 0.995 ± 0.007 | 0.000 ± 0.000 |
| clustered_redirect | aspic_v2_persistent | 0.253 | 0.996 ± 0.006 | 0.000 ± 0.000 |
| clustered_redirect | schema_gating | 0.208 | 0.995 ± 0.008 | 0.000 ± 0.000 |

## Redirect Exposure Analysis
| Dataset | Redirect exposure (schemas/problem) | Redirect exposure ratio | Clustered redirect gap (IW-gating) |
|---|---:|---:|---:|
| RuleTaker | 3.376 ± 0.000 | 0.471 ± 0.000 | 0.108 |
| ProofWriter | 3.993 ± 0.000 | 0.535 ± 0.000 | 0.004 |

## Discussion Note
ProofWriter does not show lower redirect exposure than RuleTaker (ProofWriter: 3.99, RuleTaker: 3.38), so redirect exposure alone does not explain the smaller ProofWriter redirect effect.

## Degraded Estimator Ablation
- Dataset: `ruletaker`
### clustered_negate
| Label noise | Δ Chain-Linked Error (schema_gating - instance_weighted) mean±std |
|---|---:|
| 0% | -0.132 ± 0.017 |
| 10% | -0.131 ± 0.008 |
| 20% | -0.125 ± 0.019 |
| 30% | -0.131 ± 0.016 |

### clustered_redirect
| Label noise | Δ Chain-Linked Error (schema_gating - instance_weighted) mean±std |
|---|---:|
| 0% | -0.108 ± 0.014 |
| 10% | -0.106 ± 0.016 |
| 20% | -0.100 ± 0.004 |
| 30% | -0.096 ± 0.018 |


## Estimator Notes
- Configured estimator backend: `count_fallback`
- Per-condition seed metrics saved in `results.json` under `ruletaker.estimator_metrics`, `proofwriter.estimator_metrics`.

## Synthetic Redirect-Corruption Experiment
Three-schema synthetic problems. S1 (TRANSITIVE, corruptible src_A), S2 (IMPLICATION, reliable src_B): M->Q, S3 (IMPLICATION, reliable src_B): M_wrong->not Q. Negation corruption: S1->not M (downstream schemas never fire). Redirect corruption: S1->M_wrong (S3 fires and commits not Q via cascade). Perfect estimator: src_A=0.3, src_B=0.8. 60 problems x 3 seeds.

| Model | Negation chain-error | Redirect chain-error | Δ |
|---|---:|---:|---:|
| instance_weighted | 0.000 ± 0.000 | 1.000 ± 0.000 | +1.000 |
| aspic_v1_terminal | 0.000 ± 0.000 | 0.000 ± 0.000 | +0.000 |
| aspic_v2 | 0.000 ± 0.000 | 0.444 ± 0.063 | +0.444 |
| aspic_v2_persistent | 0.000 ± 0.000 | 1.000 ± 0.000 | +1.000 |
| schema_gating | 0.000 ± 0.000 | 0.000 ± 0.000 | +0.000 |

### Redirect Chain-Length Scaling
- This table is interpreted as stability across chain lengths, not monotonic growth.
| Model | L=3 | L=4 | L=5 |
|---|---:|---:|---:|
| instance_weighted | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| aspic_v1_terminal | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| aspic_v2 | 0.444 ± 0.063 | 0.461 ± 0.021 | 0.461 ± 0.048 |
| aspic_v2_persistent | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| schema_gating | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
