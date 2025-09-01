import numpy as np
import pandas as pd
import scipy.io
import os
from scipy.stats import zscore
import yaml
from IPython.display import Audio
from icecream import ic


SOURCE_FOLDER = '/CSNG/Morici_dataset/code'
with open(f'{SOURCE_FOLDER}/params_FM.yml') as f:
    PARAMS = yaml.safe_load(f)
RATS = PARAMS['RAT_LIST']
CONTEXTS = PARAMS['CONTEXTS']
DAY_PHASES = PARAMS['DAY_PHASES']

# if it gets 'morning' or 'evening', returns corresponding context of the run, if gets context, returns context 
def chooser(keyword, sleep_mat):
    if keyword in PARAMS['CONTEXTS']:
        return keyword
    else:
        aversive_is_first = sleep_mat['NREM']['aversive'][0][0] < sleep_mat['NREM']['reward'][0][0]
        if keyword == 'morning':
            if aversive_is_first:
                return 'aversive'
            else:
                return 'reward'
        elif keyword == 'evening':
            if aversive_is_first:
                return 'reward'
            else:
                return 'aversive'
        elif keyword == 'baseline':
            return 'baseline'
        else:
            print(f'Choose context from {CONTEXTS} or {DAY_PHASES}')

def flatten(matrix):
    return [item for row in matrix for item in row]

def merge_lists_of_dicts(list1, list2, prefix):
    
    for i in range(len(list2)):
       
        for key in list2[i].keys():
            list1[i][f'{prefix}{key}'] = list2[i][key]
            
    return list1

def filter_short_epochs(l):
    d = {}
    for key in l.keys():
        d[key] = list(filter(lambda x: x[0] != x[1], l[key]))
    return d
   
def create_rat_dict(rat_N, session_list, data_folder, has_dorsal=True):
    """
    Creates dictionary with the data organised (one rat only, multiple sessions can be included).
    """
    dict_rat = {session:{} for session in session_list}
    if session_list == []:
        session_list = PARAMS['SESSIONS_LIST'][rat_N]
    print('create_rat_dict session_list: ', session_list)
    
    for session in session_list:
        dict_session = {}
        dict_session['name'] = rat_N
        
        lfp_mat = scipy.io.loadmat(f'{data_folder}/Rat{rat_N}/Rat{rat_N}-{session}/lfp.mat')
        
        ## adding dHPC data
        if has_dorsal:
            dict_session['dHPC_lfp'] = lfp_mat['dHPC'][:,1]
            dict_session['dHPC_zscore'] = scipy.stats.zscore(lfp_mat['dHPC'][:,1])
            
        ## adding vHPC data
        dict_session['vHPC_lfp'] = lfp_mat['vHPC'][:,1:].T
        dict_session['vHPC_zscore'] = scipy.stats.zscore(lfp_mat['vHPC'][:,1:], axis = 0).T
        
        ## time stamps
        dict_session['time_stamps'] = lfp_mat['vHPC'][:,0]
        
        ## adding session structure
        lfp_mat = scipy.io.loadmat(f'{data_folder}/Rat{rat_N}/Rat{rat_N}-{session}/Session_Structure.mat',simplify_cells=True)['TimeStamps']
        session_struct_dict = {}

        for activity in lfp_mat.keys():
            session_struct_dict[activity.lower()] = {}
            for context in lfp_mat[activity].keys():
                session_struct_dict[activity.lower()][context.lower()] = [lfp_mat[activity][context][0], 
                                                                          lfp_mat[activity][context][1]]
        dict_session['session_struct'] = session_struct_dict
        
        ## adding sleep structure
        sleep_struct = {}
        sleep = scipy.io.loadmat(f'{data_folder}/Rat{rat_N}/Rat{rat_N}-{session}/Sleep_Scoring_with_wake.mat',simplify_cells=True)
        sleep_struct['REM'] = filter_short_epochs(sleep['REM'])
        sleep_struct['NREM'] = filter_short_epochs(sleep['NREM'])
        sleep_struct['wake'] = filter_short_epochs(sleep['wake'])
        dict_session['sleep_struct'] = sleep_struct
        
        ## saving the given session into dict.
        dict_rat[session] = dict_session
        print(f'Session {session} for Rat_{rat_N} loaded.')
    return dict_rat

def get_index_interval(interval, timestamps):
    [start, stop] = interval
    first = np.searchsorted(timestamps, start)
    last= np.searchsorted(timestamps, stop) 
    
    return first, last

def get_zscore_epoch_rat_dict(rat_dict):
    zscored_epoch_rat_dict = {}
    counter = 0
    for session in rat_dict.keys():
        print('Session: ', session)
        timestamps_list = rat_dict[session]['time_stamps']
        zscored_epoch_rat_dict[session] = {}
        zscored_epoch_rat_dict[session]['vHPC'] = {}        
        zscored_epoch_rat_dict[session]['dHPC'] = {}        
            
        for context in CONTEXTS + DAY_PHASES:
            context_key = chooser(context, rat_dict[session]['sleep_struct'])
            for i in range(5):
                zscored_epoch_rat_dict[session]['vHPC'][context_key]= {}
            zscored_epoch_rat_dict[session]['dHPC'][context_key]= {}
                
            for sleep_phase in ['wake', 'REM', 'NREM']:
                zscored_epoch_rat_dict[session]['vHPC'][context_key][sleep_phase] = {}    
                zscored_epoch_rat_dict[session]['dHPC'][context_key][sleep_phase] = {}    
                
                zscored_epoch_rat_dict[session]['dHPC'][context_key][sleep_phase]['d'] = []    
                for i in range(5):
                    zscored_epoch_rat_dict[session]['vHPC'][context_key][sleep_phase][f'v{i}'] = []    
            
                for interval in rat_dict[session]['sleep_struct'][sleep_phase][context_key]:
                    start, stop = get_index_interval(interval, timestamps_list)
                    dHPCs = rat_dict[session]['dHPC_zscore']
                    
                    zscored_epoch_rat_dict[session]['dHPC'][context_key][sleep_phase]['d'].append(dHPCs[start:stop])
                    vHPCs = rat_dict[session]['vHPC_zscore']
                    
                    for i in range(5):
                        zscored_epoch_rat_dict[session]['vHPC'][context_key][sleep_phase][f'v{i}'].append(vHPCs[i][start:stop])
            if context == 'aversive' or context == 'reward':
                counter += len(zscored_epoch_rat_dict[session]['dHPC'][context]['REM']['d'])
    print(f'get_zscore_epoch_rat_dict {rat_dict[session]['name']} counter: ', counter)
    return zscored_epoch_rat_dict






