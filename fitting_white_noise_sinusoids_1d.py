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
output_filename = "wns1d.1e5.pickle"
store_all = False  # Set True to store all y values, all best-fitting y model values, all correl fns
                  # ...don't set with Nruns = 100000 unless you have plenty of memory!
# Number of data points
nx = 100
# Max sin, cos order m to fit - sin(2 pi m x), cos(2 pi m x) - note this relates to the maximum M in
# the Residual Entropy paper as max(M) = mmax - 1
mmax = 1 + nx//2
# Number of random runs per order m
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

# Log residual correlation power spectrum
m_lrcps = [] # Mean
s_lrcps = [] # Sample standard deviation

# Pure noise (i.e. just y) correlation power spectrum
m_lncps = [] # Mean
s_lncps = [] # Sample standard deviation

# Residual correlation function
m_rcf = [] # Mean
s_rcf = [] # Sample standard deviation

# Pure noise (i.e. just y) correlation function
m_ncf = [] # Mean
s_ncf = [] # Sample standard deviation

# Handle bulk storage if requested
if store_all:
    rcpsd = []
    ncpsd = []
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

    # Get residuals and calculate (small p) correlation power spectral density, and
    # semivariogram/correlation functions, using FFTs
    r = y - yf
    rcps = ((np.abs(np.fft.fft(r, axis=-1))**2).T / np.sum(r**2, axis=-1)).T
    ncps = ((np.abs(np.fft.fft(y, axis=-1))**2).T / np.sum(y**2, axis=-1)).T
    rc = (np.fft.ifft(rcps, axis=-1)).real
    nc = (np.fft.ifft(ncps, axis=-1)).real

    # By construction in this experiment, typical values of rcps / ncps should be ~1 for modes that
    # are not being suppressed by overfitting, thus values smaller than the machine espilon are
    # possibly suspect... So let's put in a floor of the machine epsilon
    min_log_arg = np.sys.float_info.espilon
    rcps[rcps < min_log_arg] = min_log_arg
    ncps[ncps < min_log_arg] = min_log_arg

    # Store averaged power spectrum, correlation function results
    m_lrcps.append((-np.log(rcps)).mean(axis=0))
    s_lrcps.append((-np.log(rcps)).std(axis=0))
    m_lncps.append((-np.log(ncps)).mean(axis=0))
    s_lncps.append((-np.log(ncps)).std(axis=0))
    m_rcf.append(rc.mean(axis=0))
    s_rcf.append(rc.std(axis=0))
    m_ncf.append(nc.mean(axis=0))
    s_ncf.append(nc.std(axis=0))
    if store_all:
        rcpsd.append(rcps)
        ncpsd.append(ncps)
        rcf.append(rc)
        ncf.append(nc)

# Convert to arrays and store output
m_lrcps = np.asarray(m_lrcps)
s_lrcps = np.asarray(s_lrcps)
m_lncps = np.asarray(m_lncps)
s_lncps = np.asarray(s_lncps)
m_rcf = np.asarray(m_rcf)
s_rcf = np.asarray(s_rcf)
m_ncf = np.asarray(m_ncf)
s_ncf = np.asarray(s_ncf)

output = {}
output["m_lrcps"] = m_lrcps
output["s_lrcps"] = s_lrcps

output["m_lncps"] = m_lncps
output["s_lncps"] = s_lncps

output["m_rcf"] = m_rcf
output["s_rcf"] = s_rcf

output["m_ncf"] = m_ncf
output["s_ncf"] = s_ncf

if store_all:
    yfit = np.asarray(yfit)
    output["yfit"] = yfit
    output["yall"] = yall
    rcpsd = np.asarray(rcpsd)
    rcf = np.asarray(rcf)
    ncf = np.asarray(ncf)
    output["rcpsd"] = rcpsd
    output["ncpsd"] = ncpsd
    output["rcf"] = rcf
    output["ncf"] = ncf

print("Saving results to "+output_filename)
with open(output_filename, "wb") as fout:
    pickle.dump(output, fout)
