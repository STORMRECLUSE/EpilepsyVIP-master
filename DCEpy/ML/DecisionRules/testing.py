'''

'''
from __future__ import print_function
import numpy as np


def update_decision_stats(old_stats,new_stats):
    for new_key,new_item in new_stats.items():
        if new_key in old_stats:
            old_stats[new_key].append(new_item)
        else:
            old_stats[new_key] = [new_item]

    return old_stats


def check_for_fp(inter_endtimes,inter_decisions,holdoff_per):
    '''
    Given the final decisions for an INTERICTAL period of time, as well as the times of the decisions,
    return the number of false positives and the time elapsed in the file.
    :param inter_endtimes: the time corresponding to the end of the seizure window
    :param inter_decisions: a numpy array of true or false values
    :param holdoff_per: float,int
    :return: dict
    The number of false positives in the sample, and the total time in the window in a dict.
    '''
    past_endtime = None
    false_pos = 0
    for endtime,decision in zip(inter_endtimes,inter_decisions):
        #check to see that we are indeed searching, and if we
        if decision and (not past_endtime or endtime > past_endtime + holdoff_per):
            false_pos +=1
            past_endtime = endtime

    return {'false_pos':false_pos, 'inter_time':inter_endtimes[-1]-inter_endtimes[0]}

def diagnose_seiz_file(seiz_endtimes,seiz_decisions,holdoff_per,seizure_info):
    '''
    Given the final decisions for an ICTAL period of time, return the number of false positives, as well as
    returning if the seizure has been found.
    :param seiz_endtimes:
    :param seiz_decisions:
    :param holdoff_per:
    :param seizure_info: A parameter that has start and end times
    :return:
    '''
    past_endtime = None
    false_pos = 0
    seiz_latency = np.nan
    found_seiz = False
    inter_time = max(seizure_info[0]-seiz_endtimes[0] ,0.)
    for endtime,decision in zip(seiz_endtimes,seiz_decisions):
        #check to see that we are indeed searching, and if we
        if decision and (not past_endtime or endtime > past_endtime + holdoff_per):
            #are we before, during, or after the period where we would count as a true pos.
            past_endtime = endtime
            if endtime < seizure_info[0]-holdoff_per:
                false_pos +=1
            elif seizure_info[0]-holdoff_per <= endtime<=seizure_info[1]:
                found_seiz = True
                seiz_latency = endtime -seizure_info[0]
                break
            else:
                break

    return {'seiz_latency':seiz_latency,'false_pos':false_pos,
            'inter_time':inter_time,'found_seiz':found_seiz}



