#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on March 29 8:38 PM 2025
Created in PyCharm
Created as pico_py_analysis/hex_detector_testing.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt

from HexDetector import HexDetector


def main():
    detector = HexDetector(4.3, 50, 25, np.deg2rad(5))
    detector.add_pad(0, 1, 0)
    detector.add_pad(0, -1, 0)
    detector.add_pad(0, 1, 1)
    detector.add_pad(0, 1, -1)
    detector.add_pad(0, -1, 1)
    detector.add_pad(0, -1, -1)
    detector.plot_detector(global_coords=False)
    detector.plot_detector(global_coords=True)
    print('donzo')


if __name__ == '__main__':
    main()
