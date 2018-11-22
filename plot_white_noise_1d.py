#!/usr/bin/env python
"""
plot_white_noise_1d.py
=========================
Plot examples of fitting white noise in 1D, using stored results created by the script
fitting_white_noise_1d.py
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

# if False:
#     # Show x, y results
#     plt.plot(x, y)
#     plt.plot(x, yfit)
#     plt.show()

#     # Show x, y results
#     plt.bar(x, r, width=0.01)
#     plt.show()

# if False:
#     # Plot the pure noise and residual correlation functions
#     plt.bar(np.arange(n//2) - 0.2, rcf[:n//2], width=0.4)
#     plt.bar(np.arange(n//2) + 0.2, ncf[:n//2], width=0.4)
#     plt.show()
