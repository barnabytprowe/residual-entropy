#!/usr/bin/env python
"""
plot_white_noise_1d.py
=========================
Plot examples of fitting white noise in 1D, using stored results created by the script
fitting_white_noise_1d.py
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Ready output folder
if not os.path.isdir(os.path.join(".", "plots")):
    os.mkdir(os.path.join(".", "plots"))

# Load detailed ("all") results from pickled file object
with open("wns1d.1e3.all.pickle", "r") as fin:
    rall = pickle.load(fin)
# Pull out results
yall = rall["yall"]
yfit = rall["yfit"]
rcf = rall["rcf"]
ncf = rall["ncf"]


# Single examples of fitting in 1D
# ================================
#
# Define unit interval
x = np.linspace(-.5, .5, num=yall.shape[-1], endpoint=False)

print("Sum of Square Errors (SSE) results for basic y, fit and residuals examples")
print("--------------------------------------------------------------------------")
# Loop through orders to show in the example figures - something low -> intermediate -> high
for m, fig_str in zip((1, 21, 41), ("fig1", "fig2", "fig3")):

    print("m = "+str(m))
    fig = plt.figure(figsize=(6, 4))
    plt.grid()
    plt.plot(x, yall[m, 0, :], label=r"White noise $\sim N(0, 1)$")
    plt.plot(x, yfit[m, 0, :], label=r"Best-fitting curve for $m="+str(m)+"$")
    plt.ylim(-3, 3)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig(os.path.join(".", "plots", fig_str+"a.pdf"))
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    print("SSE of y (no fit applied) = "+str(np.sum(yall[m, 0, :]**2)))
    plt.grid()
    #plt.plot(x, yall[m, 0, :], label=r"White noise $\sim N(0, 1)$")
    plt.plot(
        x, yall[m, 0, :] - yfit[m, 0, :], "r--",
        label=r"Residual from best-fitting curve for $m="+str(m)+"$")
    plt.ylim(-3, 3)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig(os.path.join(".", "plots", fig_str+"b.pdf"))
    plt.clf()
    print("SSE of fit (m="+str(m)+") residuals = "+str(np.sum((yall[m, 0, :] - yfit[m, 0, :])**2)))
    plt.close(fig)


# Residual correlation functions for single 1D examples
# =====================================================
#
# Loop through orders to show the rcf examples
mmax = 1 + ncf.shape[-1]//2
fig = plt.figure()
plt.grid()
for m in (1, 21, 41):

    plt.plot(
        rcf[m, 0, :mmax], label=r"$m="+str(m)+"$",
        ls={1: "-", 21: "--", 41: ":"}[m])
    plt.ylim(-1, 1.1)
    plt.xlabel(r"$|\Delta x|$")
    plt.ylabel("Autocorrelation function")
    plt.legend(loc=1)

plt.tight_layout()
plt.savefig(os.path.join(".", "plots", "fig4.pdf"))
plt.close(fig)
