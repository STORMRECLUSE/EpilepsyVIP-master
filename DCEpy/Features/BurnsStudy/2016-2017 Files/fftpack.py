import numpy as np

def DFT(x):
	N = len(x)
	M = []
	for i in range(N):
		M.append([])
		for j in range(N):
			value = i*j
			value *= np.pi
			value *= -2j
			value /= N
			value = np.exp(value)
			M[-1].append(value)
	result = []
	for i in range(N):
		accumulator = 0
		for j in range(N):
			accumulator += M[i][j] * x[j]
		result.append(accumulator)
	return result



# FFT not broken down yet
"""
def FFT(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd,
                               X_even + factor[N / 2:] * X_odd])
"""

def fftfreq(n, d=1.0):
	"""
	Returns the DFT sample frequencies
	"""
	f_pos = []
	f_neg = []
	denom = n * d
	if n % 2 == 0:
		half_n = n / 2
		for i in range(half_n):
			value = i * denom
			f_pos.append(value)
		for i in range(-half_n, 0):
			value = i * denom
			f_neg.append(value)
	f = f_pos + f_neg
	return f



