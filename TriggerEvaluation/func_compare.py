#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on February 11 21:13 2025
Created in PyCharm
Created as py_analysis/func_compare.py

@author: Alexandra Kallitsopoulou, akallits
"""

import numpy as np
import matplotlib.pyplot as plt

from simulation_trigger import integral


def main():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_pt = 3

    thomas_sliding_int = thomas_int(y, n_pt)
    print(f'Thomas sliding integral: {thomas_sliding_int}')
    xo, dylan_moving_int = integral(x, y, n_pt)
    dylan_moving_int = np.insert(dylan_moving_int, 0, np.zeros(n_pt-1))
    print(f'Dylan moving integral: {dylan_moving_int}')

    fig, ax = plt.subplots()
    ax.plot(x, y, label='y')
    ax.plot(x, thomas_sliding_int, label='Thomas sliding integral')
    ax.plot(x, dylan_moving_int, label='Dylan moving integral')
    ax.legend()
    plt.show()

    print('bonzo')


def thomas_int(y, n_pt):
    sliding_int = np.zeros(len(y))
    sliding_int[0] = y[0]

    for i in range(1, n_pt):
        sliding_int[i] = sliding_int[i-1] + y[i]

    for i in range(n_pt, len(y)):
        sliding_int[i] = sliding_int[i-1] + y[i] - y[i-n_pt]

    return sliding_int




if __name__ == '__main__':
    main()
