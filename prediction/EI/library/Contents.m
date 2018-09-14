% ASTRONOMICAL - Time Series Utilities
%                                 By : Eran O. Ofek
%                                 Version: September 2005
% List of MATLAB programs in the timeseries package
%
% acf       - Autocorrelation function for evenly spaced, one dimensional
%             time series.
% afoevfor  - formating AFOEV variable star data file into a MATLAB
%             variable.
% arp       - model a time series by autoregressive process of order p.
% bin_by_eye- Plot data and define binning by eye.
%             The user mark (with the mouse) the beginning
%             and end points of each bin.
%             The left and right limits of each bin defined
%             by the user are marked by cyan and red
%             dashed lines, respectively.
% binn      - Binning a set of observations by equal number
%             of observations within each bin.
%             The program returns a matrix containing
%             the mean "time", mean value and value standard
%             deviation.
%             If the number of observations is not divided by
%             the number of points in each bin without
%             a reminder, then the last remaining observations
%             will not be used.
% ccf       - Cross correlation function for two, one dimensional time series.
%             Use Edelson & Krolik binning method for not-equaly spaced series.
% ccf_o     - Cross correlation function for evenly spaced two one
%             dimensional series.
% cosbell   - cosine bell function. Generating cosine bell function
%             in the range Start to End with its inner PercentFlat part
%             as flat function.
% curvlen   - calculate the length of curve by summing the distance
%             between successive points.
% cusum     - cumulative sum (CUSUM) chart for detecting non
%             stationarity in a series mean.
% equeliz   - Given two matrices [JD, Mag, ...], select all the
%             observations in the first matrix that was made in
%             the same instant (+/-threshold) and return the in
%             each line observations from both matrices that
%             was made at the same instant.
% extracph  - Extract observation within a given phase range.
% fitexp    - LSQ fitting of exponent model, to set of data points.
% fitgauss  - linear least squars gaussian fit to data.
% fitharmo  - LSQ harmonies fitting. Fit simultaneously any number of
%             frequncies, with any number of harmonics and linear terms.
% fitharmonw- LSQ harmonies fitting, with no errors (Weights=1).
% fitlegen  - LSQ Legendre polynomial fitting.
% fitpoly   - LSQ polynomial fitting.
% fitslope  - LSQ polynomial slope fitting (no a_0 term).
% fmaxs     - Given a matrix, find local maxima (in one of
%             the columns) and return the maxima position and height.
% folding   - Folding a set of observations into a period.
%             For each observation return the phase of the
%             observation within the period.
% hjd       - Convert Julian Day (UTC) to Helicentric/Barycentric
%             Julian Day (for geocentric observer).
% minclp    - Search for periodicity in a time series,
%             using the minimum-curve-length method.
%             The program calculates the curve length for
%             each trail frequency, and return the curve
%             length as function of frequency.
% pdm       - phase dispersion minimization.
% pdm_phot  - Phase dispersion minimazation of photon arrival time series.
% periodia  - classical periodigram calculating. normalization by
%             the variance of the data.
% periodis  - calculating a power spectrum to set of observations by
%             the method of Scargle.
% periodit  - calculating power spectrum as function of time.
% perioent  - periodicity search by minimizing the entropy.
% phot_event_me - Searching periodicity in time-tagged
%             events using information entropy.
%             For each trail period, the phase-magnitude space
%             is divided into m by m cells, and the information
%             entropy is calculated.
% poisson_event - Given a vector of time-tagged events, compare
%             the delta-time between sucssive events
%             with the exponential distribution.
% polysubs  - Subtract polynomial from a data set (no errors).
% runderiv  - Calculate the runing derivative of an unevenly spaced time
%             series, with flat weighting function
%             (e.g., the slope in each window).
%             Take into account slope-error and \chi^2.
% runmean   - Calculate the runing mean of an unevenly spaced time
%             series with different weight functions and weight scheme.
% sf_interp - Interpolation with structure function error propagation.
%             The error bar in each interpolated point is
%             calculated by adding in quadrature the
%             the error of the neastest point with the
%             amplitude of the stracture function at
%             the the lag equal to the difference between
%             the interpolated point and the nearest point.
% specwin   - spectral window of a time series.
%







