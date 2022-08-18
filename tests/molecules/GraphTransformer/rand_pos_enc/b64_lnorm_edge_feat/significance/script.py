# average and std of the results of the significance test
import numpy as np
from scipy import stats

rng = np.random.default_rng()

ape_results = [0.1611, 0.1741, 0.1766, 0.1708, 0.1692]
# ape_results = [0.1 for _ in range(5)]
lape_results = [0.2543, 0.2462, 0.2471, 0.2431, 0.2835]
baseline_results = [0.2353, 0.2679, 0.2503, 0.302, 0.2228]

results = {
    'ape': ape_results,
    'lape': lape_results,
    'baseline': baseline_results
}

for key, value in results.items():
    print(key, np.mean(value), np.std(value))

print(stats.ttest_ind(ape_results, lape_results))