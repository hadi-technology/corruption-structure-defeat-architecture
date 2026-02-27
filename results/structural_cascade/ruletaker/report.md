# Structural Cascade Report

- Method: `structural cascade potential analysis`
- Dataset: `ruletaker`

## Summary
- Total schemas analyzed: `3532`
- Terminal schemas: `1811` (51.27%)
- Cascade-eligible schemas: `1721` (48.73%)
- Multi-hop cascade schemas: `1072` (30.35%)
- Avg reach depth: `1.002`
- Avg reachable schemas per schema: `1.390`

## Problem-Level Exposure
- Problems analyzed: `500`
- Mean redirect exposure ratio: `0.481`
- Mean multi-hop exposure ratio: `0.293`
- Problems with any redirect exposure: `471` (94.20%)
- Problems with any multi-hop exposure: `376` (75.20%)

### By Family
| Bucket | Total | Terminal | Cascade-eligible | Multi-hop | Avg reach depth |
|---|---:|---:|---:|---:|---:|
| IMPLICATION | 878 | 878 | 0 | 0 | 0.000 |
| TRANSITIVE | 2654 | 933 | 1721 | 1072 | 1.333 |

### By Problem Depth
| Bucket | Total | Terminal | Cascade-eligible | Multi-hop | Avg reach depth |
|---|---:|---:|---:|---:|---:|
| 3 | 1125 | 606 | 519 | 299 | 0.853 |
| 4 | 1055 | 547 | 508 | 308 | 0.944 |
| 5 | 1352 | 658 | 694 | 465 | 1.170 |

## Simulation (Corruption Counterfactual)
- Simulated schemas: `60`
- Successful simulations: `60`
- Corrupted consequent consumed downstream: `23` (38.33% of successful)
- Answer changed: `8` (13.33% of successful)
- Harmful cascades: `3` (5.00% of successful)

### Simulation Status Counts
- `ok`: 60

### Simulation Classification Counts
- `harmful-cascade`: 3
- `propagating-but-nonharmful`: 20
- `terminal-like`: 37

