from __future__ import print_function
from eegtools.io import edfplus

import numpy as np
import re

#make forwards compatible for python 3
z = zip([1,2],[3,4])
try:
    z[1] = 5
    import itertools
    zip = itertools.izip
except TypeError:
    pass

def word_in_regexp_list(text_in, regexp_list):
    for regexp in regexp_list:
        if re.match(regexp,text_in):
            return True
    return False

def edfread(filename, bad_channels = ('Events/Markers','EDF Annotations'), good_channels = None,smart_bad_channels=True,
                rec_times = None):
    '''
    Reads in edf function using the EEGTools module.

    Note: Only works in Python 2.x. I want to get a version that will run in Python 3.
    :param filename: the filename to read in a string
    :param bad_channels: an iterable that contains the names of the channels you don't want to import
    :param smart_bad_channels: use the handy dandy regular expression to get rid of more bad channels
    :param rec_time: do you only want something over a smaller period of time in the file?
                    Don't worry. Use a list of two times (in seconds) to get your groove on, and read only that data.
    :return: data, the data in an s*n numpy array,
        where s is the total number of samples
        n is the number of "good" channels
    annotations, the annotations in the EEG file in some structure
    labels, the labels for each output channel
    '''
    with open(filename,'rb') as filen:
        bad_regexp_list = []
        if smart_bad_channels:
            bad_regexp_list = [re.compile('EEG Mark\d+'), re.compile('DC\d+')]


        reader = edfplus.BaseEDFReader(filen)

        reader.read_header()

        h = reader.header
        # check important header fields
        if good_channels:
            bad_channels = [label for label in h['label'] if label not in good_channels]
        nsamples = h['n_samples_per_record']

        fs = np.asarray(h['n_samples_per_record']) / h['record_length']
        rec_len = float(h['record_length'])



        # get records
        recs = reader.records()

        if rec_times is None:
            times,signals,annotations = list(zip(*recs))
            annotations = list(annotations)
        else:
            #times = [];
            rec_times = list(rec_times)
            signals = [];annotations =[]
            container_time_start = float(np.floor(float(rec_times[0])/rec_len))*rec_len
            container_time_end = float(np.ceil(float(rec_times[1])/rec_len))*rec_len
            for time,signal,annotation in recs:

                if (container_time_start-rec_len/4)<=time <=(container_time_end + rec_len/4) :
                    #times.append(time)
                    signals.append(signal)
                    annotations.append(annotation)
                elif time > container_time_end:
                    end_time = time
                    break
            else:
                end_time = time + rec_len
                if end_time <=rec_times[1]:
                    rec_times[1] = end_time
            if not signals:
                raise ValueError('Start or stop time make data returned empty')

        #TODO: Warn if fs is not the same for those labels
        labels,indices = zip(*[(label,index) for index,label in enumerate(h['label'])
                               if (label not in bad_channels) and
                                    (not word_in_regexp_list(label,bad_regexp_list))])
        if any(fs[list(indices)] != fs[indices[0]]):
            raise ValueError('Trying to get channels sampled at different frequencies in one array')

        nsamples,fs = nsamples[indices[0]],fs[indices[0]]

        data = np.zeros((len(signals)*nsamples,len(labels)))
        for sig_ind,signal in enumerate(signals):
            start_index = sig_ind*nsamples
            stop_index = sig_ind*nsamples+nsamples
            for data_index,channel_no in enumerate(indices):
                data[start_index:stop_index,data_index]=signal[channel_no]

        del signals
        if rec_times is not None:

            start_index = int(np.floor((rec_times[0]-container_time_start)*fs +.01))

            #change start/stop indices
            stop_index =  int(np.shape(data)[0] +\
                          np.ceil((rec_times[1]-end_time)*fs-.01))
            data = data[start_index:stop_index,:]

        return data, annotations, labels

