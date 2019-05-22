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
    plt.plot(x, yall[m, 0, :], label=r"$Y \sim N(0, 1)$")
    plt.plot(x, yfit[m, 0, :], label=r"Best-fitting curve for $M="+str(m)+"$")
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
        label=r"Residual from best-fitting curve for $M="+str(m)+"$")
    plt.ylim(-3, 3)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig(os.path.join(".", "plots", fig_str+"b.pdf"))
    plt.clf()
    print("SSE of fit (M="+str(m)+") residuals = "+str(np.sum((yall[m, 0, :] - yfit[m, 0, :])**2)))
    plt.close(fig)


# Residual correlation functions for single 1D examples
# =====================================================
#
# Loop through orders to show the rcf examples
mmax = 1 + ncf.shape[-1]//2
fig, axes = plt.subplots(3, 1, sharex=True)
# Loop through 3 examples
for i, m in enumerate((1, 21, 41)):

    axes[i].plot(rcf[m, 0, :mmax], label=r"$M="+str(m)+"$")
    axes[i].set_yticks(np.arange(-0.5, 1.5, 0.5))
    axes[i].set_ylim(-1, 1.1)
    axes[i].grid()
    if i == 0: # Add a top axis with delta x units (rather than delta i)
        ax2 = axes[i].twiny()
        ax2.set_xlabel(r'$|\Delta x|$')
        ax2.set_xlim(tuple(.01 * np.asarray(axes[i].get_xlim())))
    elif i == 1:
        axes[i].set_ylabel("Residual autocorrelation")
    axes[i].legend(loc=1)

# Add x label to final subplot
axes[-1].set_xlabel(r"Lag $l$")
# Remove horizontal space between axes
plt.tight_layout()
fig.subplots_adjust(hspace=0)
# Save and close
fig.savefig(os.path.join(".", "plots", "fig4.pdf"))
plt.close(fig)


# Plot mean of all residual autocorrelations for the full sinusoidal runs
# =======================================================================
#
# Load results from pickled file object
with open("wns1d.1e5.pickle", "r") as fin:
    rse5 = pickle.load(fin)

# Build the plot output
fig, ax = plt.subplots()
rcf_plt = rse5["m_rcf"][:(mmax - 1), :mmax]

# Seismic has good contrast
im = ax.pcolor(rcf_plt, vmin=-1, vmax=+1, cmap="seismic")
fig.colorbar(im)

# Shift ticks to be at 0.5, 1.5, etc
ax.xaxis.set(ticks=np.arange(0.5, mmax, 10), ticklabels=np.arange(0, mmax, 10))
ax.yaxis.set(
    ticks=list(np.arange(0.5, mmax-1, 10))+[49.5], ticklabels=list(np.arange(0, mmax-1, 10))+[49])
ax.set_xlabel(r"Lag $l$", size="large")
ax.set_ylabel(r"$M$", size="large")
ax.set_title("Averaged residual autocorrelation")
fig.savefig(os.path.join(".", "plots", "fig5.pdf"))
plt.close(fig)

# Plot mean of all power spectra for the full sinusoidal runs
# ===========================================================
#
ps_plt = np.abs(np.fft.fft(rse5["m_rcf"]))[:(mmax - 1), :mmax]
pss_plt = (ps_plt.T / ps_plt.max(axis=1)).T
# Build the plot output
fig, ax = plt.subplots()
# Seismic has good contrast
im = ax.pcolor(pss_plt, vmin=-1, vmax=+1, cmap="seismic")
fig.colorbar(im)
# Shift ticks to be at 0.5, 1.5, etc
ax.xaxis.set(ticks=np.arange(0.5, mmax, 10), ticklabels=np.arange(0, mmax, 10))
ax.yaxis.set(
    ticks=list(np.arange(0.5, mmax-1, 10))+[49.5], ticklabels=list(np.arange(0, mmax-1, 10))+[49])
ax.set_xlabel(r"$k$", size="large")
ax.set_ylabel(r"$M$", size="large")
ax.set_title("Averaged residual power spectral signature")
fig.savefig(os.path.join(".", "plots", "fig6.pdf"))
plt.close(fig)


# Plot mean of all residual autocorrelations for the full chebyshev runs
# ======================================================================
#
# Load results from pickled file object
with open("wnc1d.1e5.pickle", "r") as fin:
    rce5 = pickle.load(fin)

# Build the plot output
fig, ax = plt.subplots(figsize=(6, 8))
rcf_plt = rce5["m_rcf"][:, :mmax]

# Seismic has good contrast
im = ax.pcolor(rcf_plt, vmin=-1, vmax=+1, cmap="seismic")
fig.colorbar(im)

# Shift ticks to be at 0.5, 1.5, etc
ax.xaxis.set(ticks=np.arange(0.5, mmax, 10), ticklabels=np.arange(0, mmax, 10))
ax.yaxis.set(
    ticks=list(np.arange(0.5, rcf_plt.shape[0], 10)),
    ticklabels=list(np.arange(0, rcf_plt.shape[0], 10)))
ax.set_xlabel(r"Lag $l$", size="large")
ax.set_ylabel(r"$M$", size="large")
ax.set_title("Averaged residual autocorrelation")
fig.savefig(os.path.join(".", "plots", "fig7.pdf"))
plt.close(fig)

# Plot mean of all power spectra for the full chebyshev runs
# ==========================================================
#
ps_plt = np.abs(np.fft.fft(rce5["m_rcf"]))[:, :mmax]
pss_plt = (ps_plt.T / ps_plt.max(axis=1)).T
# Build the plot output
fig, ax = plt.subplots(figsize=(6, 8))
# Seismic has good contrast
im = ax.pcolor(pss_plt, vmin=-1, vmax=+1, cmap="seismic")
fig.colorbar(im)
# Shift ticks to be at 0.5, 1.5, etc
ax.xaxis.set(ticks=np.arange(0.5, mmax, 10), ticklabels=np.arange(0, mmax, 10))
ax.yaxis.set(
    ticks=list(np.arange(0.5, pss_plt.shape[0], 10)),
    ticklabels=list(np.arange(0, pss_plt.shape[0], 10)))
ax.set_xlabel(r"$k$", size="large")
ax.set_ylabel(r"$M$", size="large")
ax.set_title("Averaged residual power spectral signature")
fig.savefig(os.path.join(".", "plots", "fig8.pdf"))
plt.close(fig)


# Plots of -\sum \ln(pk)
# ======================
#
# Build the plot output
fig = plt.figure(figsize=(6, 4))
plt.grid()
plt.plot(rse5["m_lrcps"][:-1, :mmax].mean(axis=-1), "b-")
plt.plot(
    rse5["m_lrcps"][:-1, :mmax].mean(axis=-1) + rse5["s_lrcps"][:-1, :mmax].mean(axis=-1), "b--")
plt.plot(
    rse5["m_lrcps"][:-1, :mmax].mean(axis=-1) - rse5["s_lrcps"][:-1, :mmax].mean(axis=-1), "b--")
plt.axhline(rse5["m_lncps"][:-1, :mmax].mean(), ls="--", color="k")
plt.show()
