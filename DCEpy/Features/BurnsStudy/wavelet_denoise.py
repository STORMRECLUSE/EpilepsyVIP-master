import math
import numpy as np
from sklearn import preprocessing

import pywt
from statsmodels.robust import mad


##############################################################################################################
"""
Applies Wavelet Denoising method to clean up signal.

Input: 
    data_chunk:: chunk of data to denoise

Output: 
    denoised_data:: Denoised data_chunk
"""
def waveletDenoise(data_chunk, wavelet_type, level, thresh_type, extension_mode):

    #calcs wavelet coeffs
    print "GETTING WAVELET COEFFICIENTS!!!"
    wave_coeffs = pywt.wavedec(data_chunk, wavelet_type, extension_mode)
    #wave_coeffs = pywt.swt(data_chunk, wavelet_type, level)
    #print "SIZE of WAVELET COEFFS:: ", np.shape(wave_coeffs)
    

    #calcs a threshold
    print "CALCULATING A THRESHOLD"
    #thresh, wavelet_coeffs = threshold_methods(thresh_type, data_chunk, wave_coeffs, level)
    thresh, wavelet_coeffs = threshold_methods(thresh_type, data_chunk, wave_coeffs, level)

    #applies threshold to wavelet coefficients
    print "APPLYING THRESHOLD"
    
    print "WAVELET COEFFS[1:]  ", wavelet_coeffs[1:]
    print "WAVELET COEFFS[  ", wavelet_coeffs, np.shape(wavelet_coeffs)

    wavelet_coeffs[1:] = (pywt.threshold(array, value = thresh, mode = 'soft') for array in wavelet_coeffs[1:])
    #wave_coeffs[1:] = (pywt.threshold(i, value = thresh, mode = 'soft') for i in wave_coeffs[1:])
    #wavelet_coeffs = (pywt.threshold(array, value = thresh, mode = 'soft') for array in wavelet_coeffs)

    #reconstructs signal w/thresholded coeffs
    print "RECONSTRUCTING SIGNAL USING WAVE COEFFS"
    #denoised_data = pywt.waverec(wave_coeffs, wavelet_type, extension_mode)
    denoised_data = pywt.waverec(wave_coeffs, wavelet_type, extension_mode)
    print "DENOISED DATA SHAPE:: ", np.shape(denoised_data), np.shape(data_chunk)
    print "Wavelet Type:: ", wavelet_type, ", Level:: ", level, ", Threshold Type:: ", thresh_type, ", Extension Mode:: ", extension_mode 
    return denoised_data


##############################################################################################################
"""
Calcs a threshold given threshold type, data chunk, wavelet coefficients, and level

Input: 
    thresh_type:: string representing type of thresholding desired
    data_chunk:: chunk of data to perform thresholding on
    wavelet_coeffs: 3D numpy array of wavelet coefficients calc from data_chunk
    level:: integer of rep how many levels desired for thresholding

Output:
    either an array of thresholds, or a single threshold value based on thresh_type
"""

def threshold_methods(thresh_type, data_chunk, wavelet_coeffs, level):

    if thresh_type == "sqtwolog":
        return sqtwolog(data_chunk, wavelet_coeffs, level)

    if thresh_type == "rigrsure":
        return rigrsure(data_chunk, wavelet_coeffs)

    if thresh_type == "heursure":
        return heursure(data_chunk, wavelet_coeffs, level)

    if thresh_type == "minimax":
        return minimax(data_chunk, wavelet_coeffs)
        
    if thresh_type == "bayes_shrink":
        return bayes_shrink(data_chunk, wavelet_coeffs)

    return 0.0    

##############################################################################################################
"""
Performs universal thresholding 

Input: 
    data_chunk:: chunk of data to perform thresholding on
    wavelet_coeffs: 3D numpy array of wavelet coefficients calc from data_chunk
    level:: integer of rep how many levels desired for thresholding

Output:
    thresh:: an array of thresholds 
    wavelet_coeffs:: returns inputed wavelet_coeffs
"""

def sqtwolog(data_chunk, wavelet_coeffs, level):
    print "PERFORMING SQTWOLOG THRESHOLDING"
    thresh = 0.0

    #universal thresh (sq_root log)

    #median absolute deviation = mad
    #theta = mad(wavelet_coeffs[-level])
    theta = mad(wavelet_coeffs[-1])

    print "WAVELET COEFFS: ", wavelet_coeffs
    print ""
    print "THETA: ", theta
    print ""
    thresh = theta * np.sqrt(2*np.log(len(data_chunk)))
    print "THRESH: ", thresh
    print ""
    return thresh, wavelet_coeffs

##############################################################################################################
"""
Performs rigrsure thresholding; calcs threshold with unbiased risk

Input: 
    data_chunk:: chunk of data to perform thresholding on
    wavelet_coeffs: 3D numpy array of wavelet coefficients calc from data_chunk
    level:: integer of rep how many levels desired for thresholding

Output:
    thresh:: a float
    wavelet_coeffs:: returns inputed wavelet_coeffs
"""

def rigrsure(data_chunk, wavelet_coeffs):
    print "PERFORMING RIGRSURE THRESHOLDING"
    thresh = 0.0
    #thresh w/unbiased risk   

    print "WAVELET COEFFICIENTS VECTOR SHAPE: ", np.shape(wavelet_coeffs)

    #square of wavelet coeffs
    print "TAKING SQUARE OF WAVELET COEFFS"
    #choosing which coeffs to select, NEW!!
    w_single = []
    for i in range(len(wavelet_coeffs[0])):
        w_single.append(wavelet_coeffs[0][i][0])
    
    wc_sq = square_of(w_single)

    #w = square_of(wavelet_coeffs[0])

    """
    #constructs single array of all wavelet coefficients
    print "CONSTRUCTING SINGLE ARRAY OF ALL SQUARE WAVELET COEFFS"
    w_list = []
    for array in w:
        w_list = np.append(w_list, array)
    """
    print "SORTING WAVELET COEFFS", np.shape(wc_sq)
    w_sorted = timsort(list(wc_sq))

           
    #finding risk value
    #constructing risk value vector
    print "FINDING RISK VECTOR"
    n = len(w_sorted)
    print "LENGTH OF Wavelet coeffs:: ", n, len(data_chunk)
    r = [((n - 2*(i + 1) + (n - (i+1))*w_sorted[i] + math.fsum(w_sorted[:(i+1)]))  / n) for i in range(n)]


    #finding risk_val == minimum value of risk vector    
    risk_val = min(r)
    print ""
    print "RISK VALUE: ", risk_val
    print ""

    coeff_at_min_risk = w_sorted[r.index(risk_val)]
    print "COEFFICIENT AT MIN RISK: ", coeff_at_min_risk
    print ""

    thresh = np.std(data_chunk) * math.sqrt(coeff_at_min_risk)
    print "THRESH: ", thresh
    print ""
    return thresh, wavelet_coeffs

##############################################################################################################
"""
Performs heursure thresholding: calcs two values, A and B, to determine whether to apply universal/sqtwolog 
thresholding vs rigrsure thresholding.

Input: 
    data_chunk:: chunk of data to perform thresholding on
    wavelet_coeffs: 3D numpy array of wavelet coefficients calc from data_chunk
    level:: integer of rep how many levels desired for thresholding

Output:
    thresh:: an array of thresholds (if sqtwolog chosen), or a float (if rigrsure chosen)
    wavelet_coeffs:: returns inputed wavelet_coeffs
"""

def heursure(data_chunk, wavelet_coeffs, level):
    print "PERFORMING HEURSURE THRESHOLDING"
    thresh = 0.0
    #Heursure thresholding

    len_wc_vector = len(wavelet_coeffs[0])
    print "LENGTH OF WC VECTOR:: ", len_wc_vector

    #squares all wavelet coefficients, then takes their sum
    wc_sq = square_of(wavelet_coeffs[0])

    wc_sq_sum = 0.0
    for array in wc_sq:
        #wc_sq_sum += math.fsum(array)
        wc_sq_sum += array[0]

    #determines whether to use sqtwolog or rigrsure
    A = (wc_sq_sum - len_wc_vector) / len_wc_vector
    B = ( ( math.log(len_wc_vector)/math.log(2) ) ** (1.5) ) * math.sqrt(len_wc_vector)

    sqtwolog_thresh = sqtwolog(data_chunk, wavelet_coeffs, level)
    rigrsure_thresh = rigrsure(data_chunk, wavelet_coeffs)

    if A > B:
        thresh = sqtwolog_thresh
        
    else:
        thresh = min(sqtwolog_thresh, rigrsure_thresh)    

    return thresh, wavelet_coeffs

##############################################################################################################
"""
Performs minimax thresholding

Input:
    data_chunk:: chunk of data to perform thresholding on
    wavelet_coeffs: 3D numpy array of wavelet coefficients calc from data_chunk
    level:: integer of rep how many levels desired for thresholding

Output:
    thresh:: float 
    wavelet_coeffs:: returns inputed wavelet_coeffs
"""

def minimax(data_chunk, wavelet_coeffs):
    print "PERFORMING MINIMAX THRESHOLDING"
    #minimax criterion thresholding
    thresh = 0.0
    #print "WAVELET COEFFS: ", wavelet_coeffs, np.shape(wavelet_coeffs)
    wc_norm = preprocessing.normalize(wavelet_coeffs[0], norm = 'l2')
    #print "WC_NORM: ", wc_norm, len(wc_norm)
    wc_norm_mult = []
    for array in wc_norm:
        wc_norm_mult.append([i/0.6745 for i in array])


    sigma = np.median(wc_norm_mult)
    print ""
    print "SIGMA: ", sigma

    #length of data_chunk
    n = len(data_chunk)

    #finds threshold
    if n > 32:
        thresh = sigma * ( 0.3936 + 0.1829 * ( math.log(n) / math.log(2) ) )
    
    print "THRESH: ", thresh
    print ""

    return thresh, wavelet_coeffs


##############################################################################################################
"""
Performs Bayes Shrink thresholding. Similar to minimax, but calcs thresh as ratio of squqre of weighted median 
of wavelet coeffs : variance of wavelet coefficients

Input:
    data_chunk:: chunk of data to perform thresholding on
    wavelet_coeffs: 3D numpy array of wavelet coefficients calc from data_chunk
    level:: integer of rep how many levels desired for thresholding

Output:
    thresh:: float 
    wavelet_coeffs:: array of zeros if sigma_x_hat is zero, and unchanged from 
                    input wavelet_coeffs otherwise
"""

def bayes_shrink(data_chunk, wavelet_coeffs):
    thresh = 0.0

    #calc sigma hat sq
    abs_wavelet_coeffs = []
    for array in wavelet_coeffs[0]:
        abs_wavelet_coeffs.append( abs(array[0]) )

    print "WAVE COEFFS SHAPE:: ", np.shape(abs_wavelet_coeffs)
    print "CALC SIGMA HAT!!!"
    sigma_hat = np.median(abs_wavelet_coeffs) / 0.6745
    print "SIGMA HAT:: ", sigma_hat

    sigma_hat_sq = sigma_hat **2
    print "SIGMA HAT SQ:: ", sigma_hat_sq

    #calc variance of wavelet_coeffs
    n = len(wavelet_coeffs[0])
    print "LENGTH OF WAVELET COEFFS[0]:: ", n
    variance_wc = 0.0

    #finding sum of square wavelet coeffs
    print "CALC VARIANCE OF WAVELET COEFFS!!!"
    for array in wavelet_coeffs[0]:
        #variance_wc += math.fsum([elem**2 for elem in array])
        variance_wc += array[0] ** 2

    variance_wc /= (n**2)
    print "VARIANCE:: ", variance_wc
    print "SIGMA HAT SQ:: ", sigma_hat_sq

    #calc sigma x hat
    sigma_x_hat = math.sqrt( max(variance_wc - sigma_hat_sq, 0) )

    #calcs threshold
    if sigma_x_hat == 0:
        #print "SIGMA X HAT IS 0!!! SETTING WAVELET COEFFS TO 0"
        thresh = max(abs_wavelet_coeffs)

        """
        #setting all wavelet coefficients to 0
        print "OG WAVELET COEFFS:: ", wavelet_coeffs, type(wavelet_coeffs)
        wavelet_coeffs = np.array(wavelet_coeffs)
        wavelet_coeffs.fill(0)
        print "NEW ZERO WAVELET COEFFS SHAPE:: ", np.shape(wavelet_coeffs), type(wavelet_coeffs)
        print "ZERO WAVELET COEFFS:: ", wavelet_coeffs
        """
        
        
    else:
        thresh = sigma_hat_sq / sigma_x_hat

    print "THRESHOLD:: ", thresh
    return thresh, list(wavelet_coeffs)
    
##############################################################################################################
"""
Input: an array of numbers
Output: square of all elements in the array
Squares all elements in an array
"""

def square_of(lst):
    return [i ** 2 for i in lst]

##############################################################################################################

"""
MergeSort Alg to order elements in an unsorted array
"""
def merge_sort(unsorted_lst):
    print "SORTING..."
    if len(unsorted_lst) <= 1:
        return unsorted_lst
# Find the middle point and devide it
    mid = len(unsorted_lst) // 2
    left_lst = unsorted_lst[:mid]
    right_lst = unsorted_lst[mid:]

    left_lst = merge_sort(left_lst)
    right_lst = merge_sort(right_lst)
    return list(merge(left_lst, right_lst))

# Merge the sorted halves

def merge(left_half,right_half):

    res = []
    while len(left_half) != 0 and len(right_half) != 0:
        if left_half[0] < right_half[0]:
            res.append(left_half[0])
            left_half.remove(left_half[0])
        else:
            res.append(right_half[0])
            right_half.remove(right_half[0])
    if len(left_half) == 0:
        res = res + right_half
    else:
        res = res + left_half
    return res    

##############################################################################################################
#timsort alg
def timsort(unsorted_lst):
    return sorted(unsorted_lst)




"""
x = [0.3e-9, 0.0000075, 8e-20, 4e-31, 2e-30, 1e-2]
print waveletDenoise(x, 'db2', 1, 'rigrsure', 'per')
test = [[1, 2, 3], [2, 3, 1], [8, 4, 3]]
test_norm = preprocessing.normalize(test, norm = "l2")
test_new = []
for array in test:
    test_new.append([i /.5 for i in array])
print test_new
"""

"""
single_data_filename = "/Volumes/Cheng/EpilepsyVIP/Data/TS057/DA001007_1-1+.edf"
print single_data_filename
start = 0
end = 5
dimensions_to_keep = choose_best_channels("TS057", seizure = 0, filename = single_data_filename)
X_chunk, _, labels = edfread.edfread(single_data_filename, rec_times=[start, end],
                                                 good_channels=dimensions_to_keep)



denoised_data = waveletDenoise(X_chunk, 'db2', 4, 'minimax', 'per')

print ""
print "OG DATA CHUNK: ", X_chunk, len(X_chunk)
print ""
print "DENOISED DATA CHUNK: ", denoised_data, len(denoised_data)
print ""
print ""
print ""
"""

