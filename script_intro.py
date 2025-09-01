import scipy.io
import elephant.signal_processing as el_sp
from scipy.stats import zscore
from scipy.signal import hilbert
import yaml
import math
import numpy as np
import numpy.ma as ma
import itertools
import os
import sys

parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)
import functions_FM as fc


DATA_FOLDER = '/CSNG/Morici_dataset/light_dataset_for_monika'
SOURCE_FOLDER = '/CSNG/Morici_dataset/code'

with open(f'{SOURCE_FOLDER}/params_FM.yml') as f:
    PARAMS = yaml.safe_load(f)



def split_wake_between_phases(_prev_in, _next_in, border):
   
    to_add = []
    if border > _prev_in[1] + 1 and border < _next_in[0]:
            to_add.append([_prev_in[1] + 1, border])
            to_add.append([border + 1, _next_in[0] - 1])
    return to_add

def save_wake_states(rat_N, session, data_folder, stripped = True):
    
    sleep_mat = scipy.io.loadmat(f'{DATA_FOLDER}/Rat{rat_N}/Rat{rat_N}-{session}/Sleep_Scoring.mat',simplify_cells=True)
    session_mat = scipy.io.loadmat(f'{DATA_FOLDER}/Rat{rat_N}/Rat{rat_N}-{session}/Session_Structure.mat',simplify_cells=True)
    nrem = sleep_mat['NREM']['all']
    rem = sleep_mat['REM']['all']
    intervals = []
    i, j = 0, 0
    
    # get all rem+nrem intervals in order
    
    while i < len(nrem) and j < len(rem):
        if nrem[i][0] < rem[j][0]:
            intervals.append(list(nrem[i]))
            i += 1
     
        else:
            intervals.append(list(rem[j]))
            j += 1
            
    if len(rem[j:]) > 0:
        intervals.append( list(rem[j:][0]))
    if len(nrem[i:]) > 0:
        intervals.append( list(nrem[i:][0]))
   
    fused_ins = []
    
    # putting together blocks of REM and NREM sleep, that don't need to be interspersed by reads
    
    prev_in = intervals[0]
    for i in range(1, len(intervals)):
        next_in = intervals[i]
        if prev_in[1] + 1 == next_in[0]:
            prev_in = [prev_in[0], next_in[1]]
        else:
            fused_ins.append(list(prev_in))
            prev_in = next_in
        if i == len(intervals) - 1:
            
            fused_ins.append(list(next_in))
        
    prev_in = prev_in
        
    # get missing intervals
    baseline_sleep_in = [session_mat['TimeStamps']['Sleep']['Baseline'][0], session_mat['TimeStamps']['Sleep']['Baseline'][1]]
    aversive_sleep_in =  [session_mat['TimeStamps']['Sleep']['Aversive'][0], session_mat['TimeStamps']['Sleep']['Aversive'][1]]
    reward_sleep_in = [session_mat['TimeStamps']['Sleep']['Reward'][0], session_mat['TimeStamps']['Sleep']['Reward'][1]]
    
    sleep_in = [min([session_mat['TimeStamps']['Sleep'][x][0] for x in ['Baseline', 'Aversive', 'Reward']]), 
                max([session_mat['TimeStamps']['Sleep'][x][1] for x in ['Baseline', 'Aversive', 'Reward']])]
    
    ## splitting sleeps in between phases
    wake_intervals_with_splits = []

    ### when stripped, we want to skip the first wake
    if stripped:
        first_index = 1
        prev_in = fused_ins[0]
    else:
        first_index = 0
        prev_in = [sleep_in[0] -1, sleep_in[0] - 1]
    
    for i in range(first_index, len(fused_ins)):
        next_in = fused_ins[i]
        b = split_wake_between_phases(prev_in, next_in, baseline_sleep_in[1])
        a = split_wake_between_phases(prev_in, next_in, aversive_sleep_in[1])
        r = split_wake_between_phases(prev_in, next_in, reward_sleep_in[1])
        
        if len(b+a+r) != 0:
            if stripped:
                prev_in = next_in
                b, a, r = [], [], []
                continue
            wake_intervals_with_splits.extend(b + a + r)
            
        else:
            new_wake = [prev_in[1] + 1, next_in[0] - 1]
            wake_intervals_with_splits.append(new_wake)
        prev_in = next_in
        
    if not stripped:
        wake_intervals_with_splits.append([prev_in[1] + 1, sleep_in[1]])

    #sort into phases
    separated_ins = {}
    all_wake_ins = wake_intervals_with_splits
    context_ints = sorted([(baseline_sleep_in, 'baseline'), 
                           (aversive_sleep_in, 'aversive'), 
                           (reward_sleep_in, 'reward')], key=lambda x: x[0][0])
    
    i = 0
    for (context_sleep_in, name) in context_ints:
        separated_ins[f'{name}_wake'] = []
        
        while i < len(all_wake_ins) and all_wake_ins[i][1] <= context_sleep_in[1]:
            in_to_append = []
            if all_wake_ins[i][0] < context_sleep_in[0]:
                in_to_append = [context_sleep_in[0], all_wake_ins[i][1]]
            elif all_wake_ins[i][1] > context_sleep_in[1]:
                in_to_append = [all_wake_ins[i][0]. context_sleep_in[1]]
            else:
                in_to_append = all_wake_ins[i] 
                
            separated_ins[f'{name}_wake'].append(in_to_append)
            i+=1
           
    # create new mat
    sleep_mat['wake'] = {}
    sleep_mat['wake']['all'] = np.array(sorted(separated_ins['baseline_wake'] 
                                               + separated_ins['aversive_wake'] 
                                               + separated_ins['reward_wake'],  key=lambda x: x[0]))
    sleep_mat['wake']['baseline'] = (separated_ins['baseline_wake'])
    sleep_mat['wake']['aversive'] = (separated_ins['aversive_wake'])
    sleep_mat['wake']['reward'] = (separated_ins['reward_wake'])


    scipy.io.savemat(f'{data_folder}/Rat{rat_N}/Rat{rat_N}-{session}/Sleep_Scoring_with_wake.mat', sleep_mat)



for rat_num in ['103', '126', '127', '128', '132', '165']:
    for session_num in [a[7:] for a in os.listdir(f'{DATA_FOLDER}/Rat{rat_num}') if a.startswith('Rat')]:
        save_wake_states(rat_num, session_num, DATA_FOLDER, stripped = True)
