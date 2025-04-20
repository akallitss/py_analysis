#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 19 22:17 2025
Created in PyCharm
Created as pico_py_analysis/lecroy_trc_reader_test

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
from lecroyparser import ScopeData


def main():
    raw_file_path = '/media/dn277127/EXTERNAL_USB/C2--Trace--00010.trc'
    data = ScopeData(raw_file_path)
    print(data)
    fig, ax = plt.subplots()
    ax.plot(data.x, data.y)

    fig2, ax2 = plt.subplots()
    for i in range(data.x.size):
        ax2.plot(data.x[i], data.y[i])
    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
