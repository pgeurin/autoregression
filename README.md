# Autoregression

Here are the tools to automatically assess and test multiple working
machine learning techniques.  

Included are two extra modules:

## Galgraphs
A library of graphing functions.

Graphs with ax take a axis from matplotlib.
Use pattern matplotlib 'fig, ax = subplots(1,1)' for best effect.

Graphs without an 'ax' input plot themselves.

The code used here HEAVILY relies upon the foundational work of Matt Drury.
This project just wouldn't be the same without it.
Pandas and matplotlib. are also foundational tools to the work.

![iris_autoregression](example_plots/iris_autoregression_02_fast.gif)

## Cleandata
Cleans pandas dataframes using modern machine learning practices.

Turn first to clean_df(). It's your friend in a world of darkness.
It detects all manner of unmentionable values and replaces them with the mean or
distinguishing feature.

![Box and Violin](example_plots/box_and_violin.png)

## Installation

A `setup.py` file is included. To install into a python environment run

```bash
pip install git+https://github.com/pgeurin/autoregression.git
```

## Versioning

0.0.1 - Working graphs.
0.0.2 - Documentation.
0.0.3 - More graphs.
0.0.4 - Cleaning. AutoRegression.
