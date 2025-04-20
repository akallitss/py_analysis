#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 19 15:47 2025
Created in PyCharm
Created as pico_py_analysis/get_waveform_test

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
import uproot

from analysis_functions import get_raw_event_data


def main():
    root_path = '/local/home/dn277127/Bureau/picosec/Run224-Pool2_TESTBEAM_tree.root'
    event_no = 2000
    data = get_raw_event_data(root_path, event_no)

    fig, ax = plt.subplots()
    ax.plot(np.array(data['amplC2']))
    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()
