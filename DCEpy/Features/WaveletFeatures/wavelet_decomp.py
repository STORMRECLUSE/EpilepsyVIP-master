import pywt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as grid
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
import scipy
from DCEpy.Features.NonlinearLib.nonlinear_grab_bag import signal_entropy
# from DCEpy.Features.Spectral_Energy.spectral_energy import



"""
Coding Log:  6/20
issue: when plotting scalogram, the ratio of the subplot is bad.
accomplished: basic computation of coeff and coeff matrix
need to do: add in computation of features and read in EEG files

"""

def normalize(X):

    """
    :param X: time series
    :return: normalized time series. Numpy array with size (number of samples, )

    """
    norm = np.linalg.norm(X)
    norm=float(norm)    #just to make sure

    if norm==0:
        return np.array(X)
    else:
        return np.array(X)/norm



def make_coeff_matrix(coeffs,level,num_samples,f_s):
    """
    :param coeffs: the wavelet transform coeffs returned by dwt. Should be a list of arrays(order: high level of decomp to low level of decomp, or low freq band to high freq band).
    :param level: the number of levels of decomposition
    :param num_samples: the total number of samples in the signal
    :return: a coeff matrix with shape (level+1, number of samples) (from top to low: high freq band to low freq band)
    some options: nonlinearality, x^2, etc

    """

    print "number of samples",num_samples


    matrix_lst=[]
    time = np.linspace(0, num_samples-1,num_samples/2)    # fit to half the number of samples over time

    # time = np.linspace(0, num_samples - 1, num_samples / 2)


    for i in range(0,level+1):

        if i!=level:    # detail coefficients
            factor = 2 ** (i+1)        # the amount of reduction compared to the total number of samples in the signal

        else:           # approximation coefficient on the last level of decomposition
            factor= 2 ** level

        coeff_array=coeffs[level-i]    # high frequency coefficients first (lowest level of decomposition)
        n=len(coeff_array)

        # unfitted curve
        x = np.linspace(0, n * factor - 1, n)

        # dumped interpolation method:
        # x=np.linspace(0,n*factor-1,n*factor) since this is adding more points
        # y=[]
        # for coeff in coeff_array:
        #     y+=factor*[coeff]
        # row_func=interp1d(x,y)


        # use interpolation to fit the size of coefficients to num_samples
        row_func=interp1d(x,coeff_array)
        row=row_func(time)
        matrix_lst.append(row)


    # take the absolute value of the entries
    return np.abs(np.array(matrix_lst))


def plot_dwt(X,f_s,coeff_mat,signal_name=None,seizure_time=None):

    """
    The function plots the signal and the scalogram of dwt.

    :param X: signal array in samples
    :param f_s: sampling fequency
    :param coeffs: matrix representing dwt coefficients in increasing frequency band order with size (number of frequency bands, number of signals)
    :param level: level of decomposition
    :param seizure_time: tuple (start of seizure, end of seizure). None if interictal file.
    :return: no returned value.
    """


    # time axis
    n=len(X)
    tot_time=1.0*n/f_s
    times=np.linspace(0,tot_time,n)


    f = plt.figure()
    f.subplots_adjust(hspace=0.2, bottom=.03, left=.07, right=.97, top=.92)

    # plot signal
    plt.subplot(2, 1, 1)
    plt.title(signal_name)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.plot(times, X, 'b')
    plt.xlim(0, times[-1])
    if seizure_time!=None:
        plt.axvline(x=seizure_time[0],color="r")
        plt.axvline(x=seizure_time[1],color="r")


    # plot scalegram
    coeff_matrix=np.log(coeff_mat)
    freq_level=coeff_matrix.shape[1]
    plt.subplot(2, 1, 2)
    interpolation = 'nearest'
    plt.title("Wavelet coefficients at level %d" % freq_level)
    plt.xlabel("time")
    plt.ylabel("decomposition level")

    # print "coeff_matrix",coeff_matrix
    # print "actual shape",coeff_matrix.shape
    # print "start scalegram!"

    plt.imshow(coeff_matrix, interpolation=interpolation,aspect='auto',extent=[0,tot_time,0,freq_level])
    if seizure_time != None:
        plt.axvline(x=seizure_time[0],color="r")
        plt.axvline(x=seizure_time[1],color="r")

    plt.show()




def discrete_wavelet_decomp(X,f_s,max_level =6,plot=False,signal_name=None,seizure_time=None, wavelet_type = 'db4'):

    """
    The function performs discrete wavelet transform on the given signal. The level of decomposition depends on the size of the signal.
    and has a maximum of 10.

    :param X: raw single channel signal data vector with shape (number of samples, 1)
    :param f_s: sampling frequency
    :param plot: Boolean value. True if plot signal and scalogram; False otherwise.
    :param signal_name: None if a title is not needed for the plot, otherwise put in a string
    :return: multilevel DWT coefficients. A list of arrays from the lowest frequency band to the hightest.

    Some options: normalize signal? (call normalize); using multiple families of wavelets; level of decomposition.

    """

    n= len(X)

    # NOTE: wavelet decomposition, wavelet db4 recommended in Subasi's survey on EEG methods
    wvlt = pywt.Wavelet(wavelet_type)

    # decomposition level should be no more than max_level
    if len(X)>2**max_level:
        #print "True!"
        coeffs = pywt.wavedec(X,wvlt,level=max_level)
        level=max_level
        #print level
    else:
        coeffs = pywt.wavedec(X,wvlt)
        level = pywt.dwt_max_level(len(X), wvlt)
        print level


    coeff_matrix=make_coeff_matrix(coeffs,level,n,f_s)
    print "shape of the coefficient matrix",coeff_matrix.shape

    if plot==True:
        plot_dwt(X,f_s,coeff_matrix,signal_name,seizure_time)

    return coeff_matrix


# single features
def window_features_from_mat(coeff_matrix,window_step = 1000,window_len = 2000,window_func = None,numpy_style = True,**func_args ):
    '''
    Computes a feature on a sliding window of the coefficient matrix
    :param coeff_matrix: the computed coeffiecient matrix above
    :param window_step:  the window step, in number samples of the original signal (automatically divided by 2 to fit dwt)
    :param window_len: the window length,        "                    "
    :param window_func: the function that computes the feature from the window
    :param numpy_style: a function that will take axis=1
    :return: Returns a matrix for which axis 0 represents time and the last axis is frequency (if applicable)
    '''
    if window_func is None:
        raise ValueError('Please give this a function to compute on each window! Example: window_func=np.mean')

    if numpy_style:
        try:
            a = func_args['axis']
        except KeyError:
            func_args['axis'] = 1

    num_windows = (np.size(coeff_matrix,1)-window_len//2)//(window_step//2) + 1
    feature_mat = []
    for window_end in range(window_len//2 - 1,np.size(coeff_matrix,1),window_step//2):
        win_feat = window_func(coeff_matrix[:,window_end+1-window_len//2:window_end],
                                **func_args)
        if numpy_style:
            win_feat = np.squeeze(win_feat)
        feature_mat.append(win_feat)
    return np.asarray(feature_mat)

