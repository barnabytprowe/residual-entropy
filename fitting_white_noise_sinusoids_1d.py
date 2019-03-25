#!/usr/bin/env python
"""
fitting_white_noise_sinusoids_1d.py
===================================
Fitting white noise with sinusoids in 1D
"""
import pickle
import numpy as np

# Setup parameters
# ----------------
output_filename = "wns1d.1e5.all.pickle"
store_all = True # Set True to store all y values, all best-fitting y model values, all correl fns
# Number of data points
nx = 100
# Max sin, cos order m to fit - sin(2 pi m x), cos(2 pi m x)
mmax = 1 + nx//2
# Number or random runs per order m
Nruns = 100000

# Script
# ------
# Define unit interval
x = np.linspace(-.5, .5, num=nx, endpoint=False)
# Pre-construct the sinusoid function arrays - for building design matrix - as these won't change
sinx = np.asarray([np.sin(2. * np.pi * float(j) * x) for j in range(0, mmax)]).T
cosx = np.asarray([np.cos(2. * np.pi * float(j) * x) for j in range(0, mmax)]).T

# Storage variables
results = [] # Output coeffs

# Residual correlation function
m_rcf = [] # Mean
s_rcf = [] # Sample standard deviation

# Pure noise (i.e. just y) correlation function
m_ncf = [] # Mean
s_ncf = [] # Sample standard deviation

# Handle bulk storage if requested
if store_all:
    rcf = []
    ncf = []
    # Pre-calculate all the ys we are going to fit
    yall = np.random.randn(mmax, Nruns, nx)
    yfit = [] # Storage for best-fitting y values

# Begin main loop - note vectorized so does all Nruns simultaneously
for im in range(mmax):

    m = 1 + im # m = order of max frequency sinusoid in the truncated Fourier series fit to the data
    print("Run for m = "+str(m)+"/"+str(mmax))

    # Following numpy SVD least-squares implementation linalg.lstsq as nicely described here:
    # https://machinelearningmastery.com/solve-linear-regression-using-linear-algebra/
    # Build design matrix
    A = np.hstack([cosx[:, :m], sinx[:, :m]])

    # Retrieve stored or generate white noise y values
    if store_all:
        y = yall[im]
    else:
        y = np.random.randn(Nruns, nx)

    # Fit using SVD
    # Note rcond=None karg call requires numpy >= 1.14.0
    coeffs = np.asarray([np.linalg.lstsq(A, yarr, rcond=None)[0] for yarr in y])
    yf = (A.dot(coeffs.T)).T # Fit values of y

    # Store results
    results.append(coeffs)
    if store_all:
        yfit.append(yf)

    # Get residuals and calculate semivariogram/correlation function using FFTs
    r = y - yf
    rc = (
        ((np.fft.ifft(np.abs(np.fft.fft(r, axis=-1))**2, axis=-1)).real).T/np.sum(r**2, axis=-1)).T
    nc = (
        ((np.fft.ifft(np.abs(np.fft.fft(y, axis=-1))**2, axis=-1)).real).T/np.sum(y**2, axis=-1)).T

    # Store correlation function results
    m_rcf.append(rc.mean(axis=0))
    s_rcf.append(rc.std(axis=0))
    m_ncf.append(nc.mean(axis=0))
    s_ncf.append(nc.std(axis=0))
    if store_all:
        rcf.append(rc)
        ncf.append(nc)

# Convert to arrays and store output
m_rcf = np.asarray(m_rcf)
s_rcf = np.asarray(s_rcf)
m_ncf = np.asarray(m_ncf)
s_ncf = np.asarray(s_ncf)

output = {}
output["m_rcf"] = m_rcf
output["s_rcf"] = s_rcf

output["m_ncf"] = m_ncf
output["s_ncf"] = s_ncf

if store_all:
    yfit = np.asarray(yfit)
    output["yfit"] = yfit
    output["yall"] = yall
    rcf = np.asarray(rcf)
    ncf = np.asarray(ncf)
    output["rcf"] = rcf
    output["ncf"] = ncf

print("Saving results to "+output_filename)
with open(output_filename, "wb") as fout:
    pickle.dump(output, fout)
