Residual-Entropy
================

This repository contains code used to generate plots and results discussed in
the paper "Residual Entropy" (see https://arxiv.org/abs/1907.03888)

The code is well-tested in a Python 2.7+ environment, but should work fine in Python 3.
Let me know by opening an Issue if I need to make tweaks to ensure this.

The two main scripts are:

* `fitting_white_noise_sinusoids_1d.py`
* `fitting_white_noise_chebyshev_1d.py`

...and the `plot_white_noise_1d.py` script creates the plots using outputs from
these two.

The scripts are simple, and duplicate each other heavily: this was a deliberate
choice to make them each as clear to follow as possible.  The required libraries
are recent versions of numpy and matplotlib.

To reproduce results in the paper:

1. Run `$ python fitting_white_noise_sinusoids_1d.py`.  This will take a few
   minutes to run and creates a large pickle file that the plotting script will
   use to make some of the early charts in the paper.

2. Open a text editor and make the following edits to the Setup parameters in
   `fitting_white_noise_sinusoids_1d.py`:

   * Set `output_filename = wns1d.1e5.pickle`
   * Set `store_all = False`
   * Set `Nruns = 100000`

   This script is then ready to run all 10^5 simulations, for which it needs to
   _not_ retain and store all Y samples and corresponding R residuals in order to
   prevent you running out of memory!

3. Run `$ python fitting_white_noise_sinusoids_1d.py` again.  This now runs the
   suite of 10^5 simulations used later in the paper, and may take significantly
   over an hour.

4. Run `$ python fitting_white_noise_chebyshev_1d.py`.  This will take multiple
   hours, so be patient.

5. Finally run `$ python plot_white_noise_1d.py` to make the figures used in the
   paper.
