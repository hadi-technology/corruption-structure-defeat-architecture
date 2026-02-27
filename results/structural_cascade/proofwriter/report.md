# Structural Cascade Report

- Method: `structural cascade potential analysis`
- Dataset: `proofwriter`

## Summary
- Total schemas analyzed: `2240`
- Terminal schemas: `1053` (47.01%)
- Cascade-eligible schemas: `1187` (52.99%)
- Multi-hop cascade schemas: `843` (37.63%)
- Avg reach depth: `1.223`
- Avg reachable schemas per schema: `1.837`

## Problem-Level Exposure
- Problems analyzed: `300`
- Mean redirect exposure ratio: `0.529`
- Mean multi-hop exposure ratio: `0.364`
- Problems with any redirect exposure: `294` (98.00%)
- Problems with any multi-hop exposure: `249` (83.00%)

### By Family
| Bucket | Total | Terminal | Cascade-eligible | Multi-hop | Avg reach depth |
|---|---:|---:|---:|---:|---:|
| IMPLICATION | 490 | 490 | 0 | 0 | 0.000 |
| TRANSITIVE | 1750 | 563 | 1187 | 843 | 1.565 |

### By Problem Depth
| Bucket | Total | Terminal | Cascade-eligible | Multi-hop | Avg reach depth |
|---|---:|---:|---:|---:|---:|
| 3 | 991 | 521 | 470 | 292 | 0.939 |
| 5 | 1249 | 532 | 717 | 551 | 1.448 |

## Simulation (Corruption Counterfactual)
- Simulated schemas: `60`
- Successful simulations: `60`
- Corrupted consequent consumed downstream: `24` (40.00% of successful)
- Answer changed: `28` (46.67% of successful)
- Harmful cascades: `9` (15.00% of successful)

### Simulation Status Counts
- `ok`: 60

### Simulation Classification Counts
- `harmful-cascade`: 9
- `propagating-but-nonharmful`: 15
- `terminal-like`: 36

