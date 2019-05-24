ResEnt
======

This repository contains code used to generate plots and results discussed
in the paper "Residual Entropy".  The code is well-tested in a Python 2.7+
environment.

The two main scripts are:

* `fitting_white_noise_sinusoids_1d.py`
* `fitting_white_noise_chebyshev_1d.py`

...and the `plot_white_noise_1d.py` script creates the plots.

The scripts are simple, and duplicate each other heavily but this was an
active choice to make them each as clear to follow as possible in a standalone
manner.  The required libraries are numpy and matplotlib.

## To reproduce results in the paper:

i.   Run `$ python fitting_white_noise_sinusoids_1d.py`

     This will take a few minutes to run and creates a large pickle file that
     the plotting script will use to make some of the early charts in the
     paper.

ii.  


