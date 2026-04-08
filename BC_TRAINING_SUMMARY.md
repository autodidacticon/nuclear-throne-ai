# BC Training Summary

## Dataset Statistics
- Transitions: 230,221
- Episodes: 17

## Training Configuration
- Device: cpu
- Epochs: 10
- Batch size: 256
- Learning rate: 0.0001
- Architecture: [256, 256]
- Activation: tanh

## Per-Epoch Metrics

| Epoch | Val Loss | Accuracy |
|-------|----------|----------|
| 1 | 1.5899 | 41.8% |
| 2 | 1.1183 | 58.1% |
| 3 | 0.8960 | 65.1% |
| 4 | 0.8011 | 65.5% |
| 5 | 0.7747 | 65.6% |
| 6 | 0.7189 | 68.9% |
| 7 | 0.7576 | 65.2% |
| 8 | 0.7556 | 66.4% |
| 9 | 0.8383 | 64.6% |
| 10 | 0.6360 | 70.4% |

## Convergence Checks

- **[PASS]** loss_convergence: best/initial = 9.41% (threshold: <80%)
- **[FAIL]** action_diversity: max dominant = 100% (threshold: <=80%)
- **[FAIL]** non_random_baseline: No eval env available — skipped
- **[PASS]** observation_coverage: 7 distinct level(s) in dataset: [1, 2, 3, 4, 5, 6, 7]

## Recommendation for Agent 06
- Suggested PPO starting learning rate: 0.0001

