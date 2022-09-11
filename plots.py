from re import I
import matplotlib.pyplot as plt
import numpy as np

pos_enc_dims = [2, 4, 8, 16, 32, 64, 128, 256]
zinc_outcomes = {
    'name': 'zinc',
    'test': [
        [0.272, 0.261, 0.279, 0.276],
        [1.606, 0.283, 1.548, 1.619],
        [0.294, 1.534, 1.560, 1.554],
        [0.295, 1.535, 1.635, 0.305],
        [0.305, 1.571, 0.264, 0.304],
        [0.290, 0.289, 13.854, 0.289],
        [7.898, 0.752, 5.665, 4.326],
        [21.124, 1.533, 0.287, 1.603]
    ],
    'val': [
        [0.321, 0.301, 0.318, 0.299],
        [1.536, 0.336, 1.440, 1.552],
        [0.311, 1.412, 1.457, 1.443],
        [0.3438, 1.43, 1.52, 0.343],
        [0.352, 1.477, 0.308, 0.340],
        [0.341, 0.340, 14.1488, 0.333],
        [8.107, 0.698, 5.707, 4.359],
        [21.592, 1.403, 0.330, 1.490]
    ]
}

cycles_outcomes =  {
    'name': 'cycles',
    'val': [
        [0.843, 0.842, 0.905, 0.837],
        [0.757, 0.898, 0.828, 0.844],
        [0.928, 0.839, 0.859, 0.848],
        [0.793, 0.848, 0.923, 0.901],
        [0.93, 0.921, 0.841, 0.934],
        [0.83, 0.931, 0.851, 0.847],
        [0.842, 0.799, 0.807, 0.855],
        [0.853, 0.86, 0.935, 0.838]
    ],
    'test': [
        [0.8387, 0.8346, 0.9045, 0.8398],
        [0.7578, 0.897, 0.8229, 0.8465],
        [0.9332, 0.8379, 0.8511, 0.8424],
        [0.7918, 0.8562, 0.9277, 0.9009],
        [0.9343, 0.924, 0.8536, 0.9325],
        [0.8519, 0.9218, 0.8486, 0.8486],
        [0.8445, 0.7978, 0.7962, 0.848],
        [0.8534, 0.8623, 0.9332, 0.8403]
    ],
}

metrics = {
    'zinc': 'MAE (lower is better)',
    'cycles': 'Acc (higher is better)'
}

for outcomes in [zinc_outcomes, cycles_outcomes]:
    name = outcomes['name']
    val = outcomes['val']
    test = outcomes['test'] 
    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].set_title(name)
    axes[0].set_xticklabels(labels=[str(x) for x in pos_enc_dims])
    axes[0].boxplot(val)
    axes[0].set_xlabel('|Q|')
    axes[0].set_ylabel(f'Validation {metrics[name]}')

    axes[1].set_title(name)
    axes[1].set_xticklabels(labels=[str(x) for x in pos_enc_dims])
    axes[1].boxplot(test)
    axes[1].set_xlabel('|Q|')
    axes[1].set_ylabel(f'Test {metrics[name]}')

    fig.tight_layout()

plt.show()
