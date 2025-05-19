# PicoAnalysis
Picosec MM Analysis Codes

# Description
This repository contains analysis codes for the PICOSEC MM detector analysis. 
This analysis is using the parameter trees that have been gereated from the the cpp_analysis repository.

The codes are designed to analyze the data collected from the beamtests (tested in April, July, August 2023 and June 2024)
and extracts the timing resolution, time walk parameterization, corrections and performs ring and circle scans along the pad surfaces.


# Requirements
- Python 3.8 or higher
- numpy
- scipy
- matplotlib
- pandas
- seaborn
- scikit-learn
- uproot
- awkward

# Installation

To install the required packages, you can use pip:
```bash
pip install -r requirements.txt
```
or conda:
```bash
conda create --name picoanalysis python=3.8
conda activate picoanalysis
conda install numpy scipy matplotlib pandas seaborn scikit-learn uproot awkward
```

# Usage
There are mainly Jupyter notebooks in the directories :
-'96_pad_analysis'
-'combined_run_analysis'
-'single_pad_analysis'
All the notebooks are self-contained and can be run independently. 
The only dependency is with the analysis_functions.py that has to be in the same directory as the notebook. 
The 96_pad_analysis directory contains the code for analyzing the 96 pads of the PICOSEC MM detector, tested in June and 
September 2024.

In the combined_run_analysis directory, the code requires the analysis_functions.py file to be in the same directory.
As well as the scan_pad_dfs directory to be in the same directory.
All the individual notebooks that are going to be combined into a single run analysis have to be in the same
directory and their results will be written in the scan_pad_dfs.

Those notebooks will be used from the combined_run_analysis notebook.

# Contributing
If you would like to contribute to this repository, please fork the repository and create a pull request with your changes.

# Contact
If you have any questions or suggestions, please feel free to contact me at:
- Email: [alexakallitsopoulou@gmail.com](mailto:alexakallitsopoulou@gmail.com)
- GitHub: [Alexandra Kallitsopoulou](akallitss)

