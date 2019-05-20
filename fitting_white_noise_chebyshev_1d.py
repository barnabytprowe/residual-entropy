#!/usr/bin/env python
"""
fitting_white_noise_chebyshev_1d.py
===================================
Fitting white noise with Chebyshev polynomials in 1D
"""
import pickle
import numpy as np
from numpy.polynomial.chebyshev import chebval

# Setup parameters
# ----------------
output_filename = "wnc1d.1e3.all.pickle"
store_all = True  # Set True to store all y values, all best-fitting y model values, all correl fns
                  # ...don't set with Nruns = 100000 unless you have plenty of memory!
# Number of data points
nx = 100
# Max sin, cos order m to fit - sin(2 pi m x), cos(2 pi m x)
mmax = 101 #1 + nx//2
# Number of random runs per order m
Nruns = 1000

# Script
# ------
# Define unit interval
x = np.linspace(-1., 1., num=nx, endpoint=True)
identn = np.eye(mmax)
# Pre-construct the Chebyshev function array - for building design matrix - as this won't change
chebyx = np.asarray([chebval(x, identrow) for identrow in identn]).T

# Storage variables
results = [] # Output coeffs

# Residual correlation power spectrum
m_rcps = [] # Mean
s_rcps = [] # Sample standard deviation

# Residual correlation function
m_rcf = [] # Mean
s_rcf = [] # Sample standard deviation

# Pure noise (i.e. just y) correlation function
m_ncf = [] # Mean
s_ncf = [] # Sample standard deviation

# Handle bulk storage if requested
if store_all:
    rcpsd = []
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
    A = chebyx[:, :m]

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

    # Store averaged power spectrum, correlation function results
    m_rcps.append(rcps.mean(axis=0))
    s_rcps.append(rcps.std(axis=0))
    m_rcf.append(rc.mean(axis=0))
    s_rcf.append(rc.std(axis=0))
    m_ncf.append(nc.mean(axis=0))
    s_ncf.append(nc.std(axis=0))
    if store_all:
        rcpsd.append(rcps)
        rcf.append(rc)
        ncf.append(nc)

# Convert to arrays and store output
m_rcps = np.asarray(m_rcps)
s_rcps = np.asarray(s_rcps)
m_rcf = np.asarray(m_rcf)
s_rcf = np.asarray(s_rcf)
m_ncf = np.asarray(m_ncf)
s_ncf = np.asarray(s_ncf)

output = {}
output["m_rcps"] = m_rcps
output["s_rcps"] = s_rcps

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
    output["rcf"] = rcf
    output["ncf"] = ncf

print("Saving results to "+output_filename)
with open(output_filename, "wb") as fout:
    pickle.dump(output, fout)
