import numpy as np
import scipy.stats
from scipy.stats import ttest_rel

from dataclasses import dataclass


@dataclass
class Model:
    name: str
    test_results: list


def main():
    best_ape = Model('ape', np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    best_lape = Model('lape', np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    baseline = Model('baseline', np.array([0.1, 0.1, 0.1, 0.1, 0.1]))

    print(ttest_rel(best_ape.test_results, baseline.test_results))




if __name__ == '__main__':
    main()