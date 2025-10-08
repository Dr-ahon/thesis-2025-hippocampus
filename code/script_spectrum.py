from scipy.signal import welch
from scipy.stats import sem
from scipy import stats
import scipy
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import yaml
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

import script_functions as fc


# DATA_FOLDER = '/CSNG/Morici_dataset/light_dataset_for_monika'
DATA_FOLDER = sys.argv[1:][0]

# SOURCE_FOLDER = '/CSNG/Morici_dataset/code'

with open(f'/params_FM.yml') as f:
    PARAMS = yaml.safe_load(f)
RATS = PARAMS['RAT_LIST']
CONTEXTS = PARAMS['CONTEXTS']
DAY_PHASES = PARAMS['DAY_PHASES']
RAT_N = '103'
SESSION = '220715'
SUPTITLE_SETTING = {'color': 'b', 'y': 0.95, 'fontsize': 20, 'weight': 'bold'}
GREEN = '\033[92m'
RED = '\033[91m'
COLOR_END = '\033[0m'

HOME_PLOTS_FOLDER_NAME = 'plots'
SPECTRUM_PLOTS_FOLDER_NAME = 'spectrum_plots'

current_directory = os.getcwd()

HOME_PLOTS_PATH = os.path.join(current_directory, fr'{HOME_PLOTS_FOLDER_NAME}')
if not os.path.exists(HOME_PLOTS_PATH):
    os.makedirs(HOME_PLOTS_PATH)

SPECTRUM_PLOTS_PATH = os.path.join(HOME_PLOTS_PATH, fr'{SPECTRUM_PLOTS_FOLDER_NAME}')
if not os.path.exists(SPECTRUM_PLOTS_PATH):
    os.makedirs(SPECTRUM_PLOTS_PATH)


### DEFINITIONS ###


## helper functions and data loading functions

def prepare_and_merge_before_plotting(phase, session, d, mode, rat_dict):

    data = []
    if mode == 'all':
        for context in CONTEXTS + DAY_PHASES:
            context_key = fc.chooser(context, rat_dict[session]['sleep_struct'])
            
            curve = d[session]['vHPC'][context_key][phase]
            curve['d'] = d[session]['dHPC'][context_key][phase]['d']
            data.append(curve)

    elif mode == 'merge_channels':
        for context in CONTEXTS + DAY_PHASES:
            context_key = fc.chooser(context, rat_dict[session]['sleep_struct'])
            mean_pow_array = [d[session]['vHPC'][context_key][phase][f'v{i}']['pow_avg'] for i in range(5)]
            
            new_mean = np.mean(mean_pow_array, axis=0)
            new_sem = sem(mean_pow_array,axis=0)
            f = d[session]['vHPC'][context_key][phase]['v0']['f']
           
            new_dict = {'v': {'f': f, 'pow_array': mean_pow_array, 'pow_avg': new_mean, 'pow_sem': new_sem}}
            new_dict['d'] = d[session]['dHPC'][context_key][phase]['d']
            data.append(new_dict)
    
    return  data

def get_zscore_epoch_rat_dict(rat_dict):
    
    zscored_epoch_rat_dict = {}
    for session in rat_dict.keys():
        print('Session: ', session)
        timestamps_list = rat_dict[session]['time_stamps']
        zscored_epoch_rat_dict[session] = {}
        zscored_epoch_rat_dict[session]['vHPC'] = {}        
        zscored_epoch_rat_dict[session]['dHPC'] = {}        
            
        for context in CONTEXTS + DAY_PHASES:
            context_key = fc.chooser(context, rat_dict[session]['sleep_struct'])
            for i in range(5):
                zscored_epoch_rat_dict[session]['vHPC'][context_key]= {}
            zscored_epoch_rat_dict[session]['dHPC'][context_key]= {}
                
            for sleep_phase in ['wake', 'REM', 'NREM']:
                zscored_epoch_rat_dict[session]['vHPC'][context_key][sleep_phase] = {}    
                zscored_epoch_rat_dict[session]['dHPC'][context_key][sleep_phase] = {}    
                
                for i in range(5):
                    zscored_epoch_rat_dict[session]['vHPC'][context_key][sleep_phase][f'v{i}'] = []    
                zscored_epoch_rat_dict[session]['dHPC'][context_key][sleep_phase]['d'] = []    
            
                for interval in rat_dict[session]['sleep_struct'][sleep_phase][context_key]:
                    start, stop = get_index_interval(interval, timestamps_list)
                    dHPCs = rat_dict[session]['dHPC_zscore']
                    
                    zscored_epoch_rat_dict[session]['dHPC'][context_key][sleep_phase]['d'].append(dHPCs[start:stop])
                    vHPCs = rat_dict[session]['vHPC_zscore']
                    
                    for i in range(5):
                        zscored_epoch_rat_dict[session]['vHPC'][context_key][sleep_phase][f'v{i}'].append(vHPCs[i][start:stop])
           
    return zscored_epoch_rat_dict

def get_avg_spectra_of_epoch_list(blocks,nperseg=1024,samp_f=1250):
    
    """
    returns f from the Welch method
    returns array with PSD for all blocks
    return mean of PSD and Standard error of the mean
    """
    
    power_list = []
    for block in blocks:
        f, p = welch(block,nperseg=nperseg,fs=samp_f)
        power_list.append(p)
    if len(np.unique([pow.shape[0] for pow in power_list]))>1:
        print([pow.shape[0] for pow in power_list])
        
        power_list = [pow for pow in power_list if len(pow)==len(power_list[0])]
        print('power_list', len(power_list))
        
        print('Some intervals were too short to compute the spectrum.')
    pow_array = np.vstack(power_list)
    pow_avg = np.mean(pow_array,axis=0)
    
    pow_sem = sem(pow_array,axis=0)
    return {'f':f, 'pow_array':pow_array, 'pow_avg':pow_avg, 'pow_sem':pow_sem}

def get_all_average_spectra(zscored_epoch_rat_dict, rat_dict):
    
    new_dict = {session: 
                {side: 
                 {context: 
                  {sleep_phase: 
                   {} for sleep_phase in ['REM', 'NREM', 'wake']} for context in CONTEXTS + DAY_PHASES} for side in ['dHPC', 'vHPC']} for session in zscored_epoch_rat_dict.keys()}
    
    for session in zscored_epoch_rat_dict.keys():
        for context in CONTEXTS + DAY_PHASES:
            for sleep_phase in ['REM', 'NREM', 'wake']:
                context_key = fc.chooser(context, rat_dict[session]['sleep_struct'])

                zscores_d = zscored_epoch_rat_dict[session]['dHPC'][context_key][sleep_phase]['d'] 

                
                new_dict[session]['dHPC'][context][sleep_phase]['d'] = get_avg_spectra_of_epoch_list(zscores_d,nperseg=1024,samp_f=1250)
                
                for i in range(5): 
                    zscores_v_chan = zscored_epoch_rat_dict[session]['vHPC'][context_key][sleep_phase][f'v{i}']
                    new_dict[session]['vHPC'][context][sleep_phase][f'v{i}'] = get_avg_spectra_of_epoch_list(zscores_v_chan,nperseg=1024,samp_f=1250)
           
    return new_dict

def get_all_spectrum_plots_sessions(rat_N, _rat_dict = None, mode = 'all'):
    sessions = PARAMS['SESSIONS_LIST'][rat_N]
    if _rat_dict == None:
        rat_dict = fc.create_rat_dict(rat_N,sessions,DATA_FOLDER,has_dorsal=True)
    else:
        rat_dict = _rat_dict
        
    x = get_zscore_epoch_rat_dict(rat_dict)
    d = get_all_average_spectra(x, rat_dict)
    REM_avg_spectra = [{} for i in range(len(CONTEXTS + DAY_PHASES))]
    NREM_avg_spectra = [{} for i in range(len(CONTEXTS + DAY_PHASES))]
    wake_avg_spectra = [{} for i in range(len(CONTEXTS + DAY_PHASES))]
    for session in sessions:
       
        REM_next_session_list =  prepare_and_merge_before_plotting('REM', session, d, mode, rat_dict) 
        NREM_next_session_list =  prepare_and_merge_before_plotting('NREM', session, d, mode, rat_dict) 
        wake_next_session_list =  prepare_and_merge_before_plotting('wake', session, d, mode, rat_dict) 
        for i in range(5):   
                REM_avg_spectra = fc.merge_lists_of_dicts(list1 = REM_avg_spectra, list2 = REM_next_session_list, prefix = session)
                NREM_avg_spectra = fc.merge_lists_of_dicts(list1 = NREM_avg_spectra, list2 = NREM_next_session_list, prefix = session)
                wake_avg_spectra = fc.merge_lists_of_dicts(list1 = wake_avg_spectra, list2 = wake_next_session_list, prefix = session)
                
    return REM_avg_spectra, NREM_avg_spectra, wake_avg_spectra

def get_vd_indices(side, keys, mode):
    if mode == 'all':
        pattern = r'.*{}$'.format(side)
    elif mode == 'merge_channels': 
        pattern = r'.*{}[0-9]?$'.format(side)
    output = list(filter(lambda key: re.search(pattern, key), keys))  
    
    return output

def plot_spectra(title, 
                 REM_avg_spectra, 
                 NREM_avg_spectra, 
                 wake_avg_spectra, 
                 areas_short, 
                 min_f=1,
                 max_f=100,
                 colors={'v0':'orange','v2':'blue','v3':'green','v4':'yellow','v5':'purple','v1':'gray', 'v': 'blue', 'd': 'red'},
                 scale='linear'):
   
    sns.set_theme(style="darkgrid")
    sns.set_context("notebook")
    fig, ax = plt.subplots(5, 3,sharey=False)
    fig.set_figwidth(12)
    fig.set_figheight(12)
            
    plt.suptitle(f'{title}, freqs {min_f} to {max_f}')
    
    spectras = [wake_avg_spectra, REM_avg_spectra, NREM_avg_spectra]
    spectras_names = ['wake', 'REM', 'NREM']
    context_names = ['baseline', 'aversive', 'reward', 'morning', 'evening']

    for phase_index in range(len(spectras_names)):
        
        for context_index in range(len(context_names)):
            
            for area in areas_short:
                avg_spectra = spectras[phase_index][context_index]
                if area in avg_spectra.keys():
                    dict_area = avg_spectra[area]
                    f = dict_area['f']
                    
                    pow_avg = dict_area['pow_avg'] 
                    pow_sem = dict_area['pow_sem'] 
                    f_mask = (f>min_f) & (f<max_f)
                    f_masked = f[f_mask]
                    mean_masked = pow_avg[f_mask]
                    sem_masked = pow_sem[f_mask]
                    
                    if scale=='linear':
                        ax[context_index, phase_index].plot(f_masked,mean_masked,color=colors[area],alpha=0.8,label=area)
                    elif scale=='log_y':
                        ax[context_index, phase_index].semilogy(f_masked,mean_masked,color=colors[area],alpha=0.8,label=area)
                    elif scale=='log_x_y':
                        ax[context_index, phase_index].loglog(f_masked,mean_masked,color=colors[area],alpha=0.8,label=area)
                    else:
                        print('Wrong scale.')
                        return
                    ax[context_index, phase_index].fill_between(f_masked,mean_masked-sem_masked,mean_masked+sem_masked,color=colors[area],alpha=0.2)
                    ax[context_index, phase_index].legend()

                    for (f1, f2, col) in [(0,4, 'purple'), (4, 12, 'orange'), (30,47, 'yellow'), (53, 80, 'green')]:
                        
                        ax[context_index, phase_index].axvspan(f1, f2, color=col, alpha=0.05)  # alpha=1.0 makes it fully opaque

                else:
                    ax[context_index, phase_index].axis('off')

                ax[context_index, phase_index].set_title(f'{spectras_names[phase_index]} {context_names[context_index]}')
                ax[context_index, 0].set_ylabel('PSD\n[norm.u.]')
                ax[-1, phase_index].set_xlabel('Hz')
                
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(SPECTRUM_PLOTS_PATH, 'spectrum_plot'))

    return

def get_index_interval(interval, timestamps):
    [start, stop] = interval
    first = np.searchsorted(timestamps, start)
    last= np.searchsorted(timestamps, stop) 
    
    return first, last

def merge_dicts_keep_keys(dict1, dict2):
    merged_dict = {}
    for context_name in dict1.keys():
        merged_dict[context_name] = {}
        for channel_key in dict1[context_name].keys():
            merged_dict[context_name][channel_key] = dict1[context_name][channel_key] + dict2[context_name][channel_key]
    
    return merged_dict

## fig 17

def prepare_and_merge_all_rats(phase_d, name, mode):
    return_list = []
    for context_index in range(len(phase_d)):
        if mode == 'merge_channels':
            v_keys =  get_vd_indices('v', phase_d[context_index].keys(), mode)
            mean_v_pow_array = [phase_d[context_index][key]['pow_avg'] for key in v_keys]
            new_v_mean = np.mean(mean_v_pow_array, axis=0)
            new_v_sem = sem(mean_v_pow_array,axis=0)
            f = phase_d[context_index][v_keys[0]]['f']
            new_dict = {'v': {'f': f, 'pow_array': mean_v_pow_array, 'pow_avg': new_v_mean, 'pow_sem': new_v_sem}}
        
        elif mode == 'all':
            new_dict = {f'v{num}': {} for num in range(5)}
            
            for v_key in new_dict.keys():
                v_keys =  get_vd_indices(v_key, phase_d[context_index].keys(), mode)
                mean_v_pow_array = [phase_d[context_index][key]['pow_avg'] for key in v_keys]
                new_v_mean = np.mean(mean_v_pow_array, axis=0)
                new_v_sem = sem(mean_v_pow_array,axis=0)
                f = phase_d[context_index][v_keys[0]]['f']
                new_dict[v_key] = {'f': f, 'pow_array': mean_v_pow_array, 'pow_avg': new_v_mean, 'pow_sem': new_v_sem}

        d_keys =  get_vd_indices('d', phase_d[context_index].keys(), mode)
        mean_d_pow_array = [phase_d[context_index][key]['pow_avg'] for key in d_keys]
        new_d_mean = np.mean(mean_d_pow_array, axis=0)
        new_d_sem = sem(mean_d_pow_array,axis=0)
        
        f = phase_d[context_index][d_keys[0]]['f']
        new_dict['d'] = {'f': f, 'pow_array': mean_d_pow_array, 'pow_avg': new_d_mean, 'pow_sem': new_d_sem}
        return_list. append(new_dict)
    return return_list

def print_all_spectrum_plots_rats(Hz = 100, mode = 'all', separate_rats = False, print_plots = True):
    REM_avg_spectra = [{} for i in range(len(CONTEXTS + DAY_PHASES))]
    NREM_avg_spectra = [{} for i in range(len(CONTEXTS + DAY_PHASES))]
    wake_avg_spectra = [{} for i in range(len(CONTEXTS + DAY_PHASES))]

    for rat in RATS:
        print(rat)
        if mode == 'all':
            REM_rat_list, NREM_rat_list, wake_rat_list = rat_datasets_all_channels[rat]
        else:
            REM_rat_list, NREM_rat_list, wake_rat_list = rat_datasets_merged_channels[rat]
            
        
        if separate_rats:
            REM_rat_list = prepare_and_merge_all_rats(REM_rat_list, 'REM', mode)
            NREM_rat_list = prepare_and_merge_all_rats(NREM_rat_list, 'NREM', mode)
            wake_rat_list = prepare_and_merge_all_rats(wake_rat_list, 'wake', mode)
            
            title = f'{rat} rat average'
            areas_short = list(REM_rat_list[0].keys())
           
            plot_spectra(title,
                 REM_rat_list,
                 NREM_rat_list,
                 wake_rat_list, 
                 areas_short,
                 scale = 'log_y',
                 max_f = Hz)
        
        else:
            
            REM_avg_spectra = fc.merge_lists_of_dicts(REM_avg_spectra, REM_rat_list, rat)
            NREM_avg_spectra = fc.merge_lists_of_dicts(NREM_avg_spectra, NREM_rat_list, rat)
            wake_avg_spectra = fc.merge_lists_of_dicts(wake_avg_spectra, wake_rat_list, rat)

    if not separate_rats:
        
        REM_avg_spectra = prepare_and_merge_all_rats(REM_avg_spectra, 'REM', mode)
        NREM_avg_spectra = prepare_and_merge_all_rats(NREM_avg_spectra, 'NREM', mode)
        wake_avg_spectra = prepare_and_merge_all_rats(wake_avg_spectra, 'wake', mode)
        
        title = f'all rats average'
        areas_short = list(REM_avg_spectra[0].keys())
        plot_spectra(title,
             REM_avg_spectra,
             NREM_avg_spectra,
             wake_avg_spectra, 
             areas_short,
             scale = 'log_y',
             max_f = Hz)
             
## fig 18, 19

def apply_band_mask(rat_dict, freq, freq_limits_dict, baseline_normalize = False, ME_normalize = False, sessions = None, rat = None, channel_names = []):
  
    context_names = ['baseline', 'aversive', 'reward', 'morning', 'evening']
    binned_data = {session: 
                   {context_names: {}
                    for context_names in context_names} 
                   for session in sessions}
    ME_normalizing_factors = {f'{side_prefix}_{freq}': {'evening': [], 
                                         'morning': []} for side_prefix in channel_names}

    for session_prefix in sessions:    
      
        b_normalizing_factors = {}
        
        
        for context_name_index in range(0, len(context_names)):
            context_name = context_names[context_name_index]
            for plot_line_name in channel_names:
                binned_data[session_prefix][context_name][f'{plot_line_name}_{freq}'] = []
                f = rat_dict[context_name_index][session_prefix][plot_line_name]['f'].copy()
                f_freq_mask = (f>freq_limits_dict[freq]['min']) & (f<freq_limits_dict[freq]['max']) # array of bools
                f_freq = f[f_freq_mask]
                freq_dict = rat_dict[context_name_index][session_prefix][plot_line_name].copy()
                freq_dict['filtered_freq'] = f_freq
                freq_pow_array = [] # array of REMs, one value for one epoch and freq (more for the whole band)

                
                for arr in freq_dict['pow_array']:
                    freq_pow_array.append(np.mean(arr[f_freq_mask]))
               
                if baseline_normalize:
                    
                    if context_name == 'baseline':
                        b_normalizing_factors[f'{plot_line_name}_{freq}'] = np.mean(freq_pow_array)
                    
                    freq_pow_array = [a / b_normalizing_factors[f'{plot_line_name}_{freq}'] for a in freq_pow_array]
                        
                if ME_normalize:

                    if context_name == 'morning':
                        ME_normalizing_factors[f'{plot_line_name}_{freq}']['morning'] += freq_pow_array
                    elif context_name == 'evening':
                        ME_normalizing_factors[f'{plot_line_name}_{freq}']['evening'] += freq_pow_array
              
                binned_data[session_prefix][context_name][f'{plot_line_name}_{freq}'] += freq_pow_array
                
    return_data = {context_names: {f'{plot_line_name}_{freq}': [] for plot_line_name in channel_names} for context_names in context_names} 
    
    if ME_normalize:
        for plot_line_name in channel_names: 
            ME_normalizing_factors[f'{plot_line_name}_{freq}']['morning'] = np.mean(ME_normalizing_factors[f'{plot_line_name}_{freq}']['morning'])
            ME_normalizing_factors[f'{plot_line_name}_{freq}']['evening'] = np.mean(ME_normalizing_factors[f'{plot_line_name}_{freq}']['evening'])
    
    for session in binned_data.keys():
    
        if ME_normalize:
            sleep_mat = scipy.io.loadmat(f'{DATA_FOLDER}/Rat{rat}/Rat{rat}-{session}/Sleep_Scoring_with_wake.mat',simplify_cells=True)
            
            for context_name in binned_data[session].keys():
                if fc.chooser('morning', sleep_mat) == context_name or context_name == 'morning':
                    for plot_freq_line_name in binned_data[session]['baseline'].keys():
                        return_data[context_name][plot_freq_line_name] += [a / ME_normalizing_factors[plot_freq_line_name]['morning'] for a in binned_data[session][context_name][plot_freq_line_name]]
                elif fc.chooser('evening', sleep_mat) == context_name or context_name == 'evening':
                    for plot_freq_line_name in binned_data[session]['baseline'].keys():
                        return_data[context_name][plot_freq_line_name] += [a / ME_normalizing_factors[plot_freq_line_name]['evening'] for a in binned_data[session][context_name][plot_freq_line_name]]    
        else:
          
            return_data = merge_dicts_keep_keys(return_data, binned_data[session])
        
  
    return return_data
        
def spectral_powers(test_name = 'Kruskal', precomputed_data = False, print_violin = True, baseline_normalize = False, ME_normalize = False, rats = RATS, sessions = None, contexts_to_plot = None, mode = 'all'):
    if sessions:
        if len(rats) > 1:
            print('More rats with set sessions do not make sense')
        else:
            sessions = PARAMS['SESSIONS_LIST'][rats[0]] if sessions == None else sessions

    freq_limits_dict = {'delta': {'min': 0,
                                  'max' : 4},
                        'theta': {'min': 4,
                                  'max' : 12},
                        'gamma': {'min': 30,
                                  'max' : 47},
                        'high gamma': {'min' : 53,
                                       'max' : 80}
                        }
    context_names = ['baseline', 'aversive', 'reward', 'morning', 'evening']

    heatmap_data = {rat: {'aversive': {},
                          'reward': {},
                          'baseline': {}}
                    for rat in rats}
    
    rat_datasets = {}
    
    if mode == 'all':
        if len(rats) > 1:
            print('It does not make sense for more than one rat')
            return
        side_prefixes = ['v0', 'v1', 'v2', 'v3', 'v4', 'd']
        rat_datasets = rat_datasets_all_channels
    
    elif mode == 'merge_channels':
        if precomputed_data == False:
            side_prefixes = ['v', 'd']
            
            for rat_N in RATS: 
                rat_datasets[rat_N] = get_all_spectrum_plots_sessions(rat_N, _rat_dict = None, mode = 'merge_channels')
        else:
            rat_datasets = rat_datasets_merged_channels
            side_prefixes = ['v', 'd']

    elif type(mode) == list:
        side_prefixes = mode
        rat_datasets = rat_datasets_all_channels

            
    else:
        print('invalid mode')
        return
    if len(rats) == 5:
        rats_title = 'all rats'
    else:
        rats_title = rats
    freq_num = len(list(freq_limits_dict.keys()))
    title = f'Spectral powers of REM epochs in frequency bands of interest of {rats_title}'
    subtitle = f'baseline norm = {f'{baseline_normalize}'.upper()}       M/E norm = {f'{ME_normalize}'.upper()}'
    sns.set_theme(style="darkgrid")
    sns.set_context("notebook")
    fig, ax = plt.subplots(freq_num, len(side_prefixes), sharey=False)
    
    if baseline_normalize:
        if contexts_to_plot == 'BARME':
            contexts_to_plot = 'ARME'
        elif contexts_to_plot == 'BAR':
            contexts_to_plot = 'AR'
    
    fig.set_figwidth(15)
    fig.set_figheight(4 * freq_num + 2)
    suptitle = plt.suptitle(title)
    suptitle.set(**SUPTITLE_SETTING)
    fig.text(0.5, 0.92, subtitle, va= 'top', ha='center', fontsize=15)
    plt.subplots_adjust(hspace=0.3)
    
    for (index, freq) in enumerate(freq_limits_dict):
        
        binned_data = {context_name: 
                       {f'{side_prefix}_{freq}' : [] 
                                      for side_prefix in side_prefixes} 
                       for context_name in context_names}

        for rat_N in rats:
            if len(rats) > 1 or sessions == None:
                sessions = PARAMS['SESSIONS_LIST'][rat_N]

            rem, nrem, wake = rat_datasets[rat_N]

            # list of dicts of the length of CONTEXTS
            session_data_to_normalize = [{session: 
                                          {side_prefix : {} for side_prefix in side_prefixes} 
                                          for session in  sessions} 
                                         for i in range(len(context_names))]      

            for session_prefix in sessions:
               
                # normalization of one chunk of data (one session)
                for context_ind in range(len(context_names)):
                    for side_prefix in side_prefixes:
                        session_data_to_normalize[context_ind][session_prefix][side_prefix] = rem[context_ind][session_prefix + side_prefix]
            
            binned_session_data = apply_band_mask(session_data_to_normalize, 
                                                  freq, 
                                                  freq_limits_dict, 
                                                  baseline_normalize = baseline_normalize, 
                                                  ME_normalize = ME_normalize,
                                                  rat = rat_N,
                                                  sessions = sessions,
                                                  channel_names = side_prefixes
                                                 )
            binned_data = merge_dicts_keep_keys(binned_data, binned_session_data)
            for context in heatmap_data[rat_N].keys():
                heatmap_data[rat_N][context].update(binned_data[context])  
       
        #####
        side_freqs = [f'{side_prefix}_{freq}' for side_prefix in side_prefixes]
        max_value = 0
        min_value = np.inf
        for (side_freq_index, side_freq) in enumerate(side_freqs):
            
            if contexts_to_plot == 'ME':
                keys = ['morning', 'evening']                
                pairs = [('morning', 'evening')]
                if ME_normalize:
                    print('does not make much sense to print ME normalized morning and evening')
            elif contexts_to_plot == 'BAR':
                keys = ['aversive', 'reward', 'baseline']                
                pairs = [('aversive', 'reward'), ('baseline', 'reward'), ('aversive', 'baseline')]
            elif contexts_to_plot == 'AR' or ME_normalize:
                keys = ['aversive', 'reward']
                pairs = [('aversive', 'reward')]
            elif contexts_to_plot == 'ARME':
                keys = ['aversive', 'reward', 'morning', 'evening']                
                pairs = [('aversive', 'reward'),('morning', 'evening')]
            elif contexts_to_plot == 'BARME':
                keys = ['aversive', 'reward', 'baseline', 'morning', 'evening']                
                pairs = [('aversive', 'reward'),('morning', 'evening'), ('baseline', 'reward'), ('baseline', 'evening'), ('baseline', 'morning'), ('baseline', 'aversive')]
            else:
                print('invalid contexts_to_plot')
                return

            plotting_data = {key: binned_data[key][side_freq] for key in keys}
            
            max_len = max([len(plotting_data[key]) for key in plotting_data.keys()])
            this_max = max([max(plotting_data[key]) for key in plotting_data.keys()])
            max_value = this_max if this_max > max_value else max_value
            this_min = min([min(plotting_data[key]) for key in plotting_data.keys()])
            min_value = this_min if this_min < min_value else min_value
           
            # adding NaNs
            for key in plotting_data.keys():
                plotting_data[key] = [*plotting_data[key], *[np.nan] * (max_len-len(plotting_data[key]))]

            if not print_violin:
                plt.close()

            df = pd.DataFrame(plotting_data)
            df_melted = df.melt(var_name='Group', value_name='Values')
            with sns.plotting_context('notebook'):
                plotting_parameters = {'data' : df_melted, 
                                       'x' : 'Group', 
                                       'y' : 'Values', 
                                       'hue' : 'Group', 
                                       'palette' : {'aversive': 'r', 'reward': 'y', 'morning': 'c', 'evening': 'm', 'baseline': 'g'},
                                       'linewidth' : 0.8,
                                       'inner' : 'box',
                                       'edgecolor' : (0, 0, 0, 0.2)
                                      }
                
                sns.set_theme(style="darkgrid")
                sns.violinplot(**plotting_parameters, ax = ax[index][side_freq_index])
                ax[index][side_freq_index].set_title(side_freq, fontweight='bold', fontsize = 16)
                ax[index][side_freq_index].set_ylabel(None)
                ax[index][side_freq_index].set_xlabel(None)
                ax[index][side_freq_index].xaxis.set_tick_params(labelsize = 17)
                ax[index][side_freq_index].yaxis.set_tick_params(labelsize = 13)
                ax[index, 0].set_ylabel('PSD [norm.u.]')
                
                annotator = Annotator(ax[index][side_freq_index], pairs, **plotting_parameters)
                annotator.configure(test=test_name, 
                                    loc='inside', 
                                    text_format='star', 
                                    show_test_name=True, 
                                    line_height=0.01, 
                                    fontsize=12, 
                                    color='gray', 
                                    verbose=False)           
                annotator.apply_and_annotate()
                
        for (i, name) in enumerate(side_freqs):
            plt.setp(ax[index][i], ylim=(min_value - 0.15 * max_value, max_value + 0.15 * max_value))
            
    if print_violin:
        plt.savefig(os.path.join(SPECTRUM_PLOTS_PATH, 'spectrum_powers.png'))
        
    else:
        plt.close()
        return heatmap_data

## fig 20


def prepare_for_power_correlation_plotting(norms, channels):
    freqs = ['delta', 'theta', 'gamma', 'high gamma']
    corr_dict = {freq: 
                 {context: [] 
                  for context in ['aversive', 'reward', 'baseline', 'morning', 'evening']}
                   for freq in freqs} 
                 
    all_corr_counter = 0
    sig_corr_counter = 0
    for rat in RATS:
        print(rat)
        if norms['ME_normalize']:
            contexts_to_plot = 'ARME'
            contexts = ['aversive', 'reward']
        else:
            contexts_to_plot = 'BARME'
            contexts = ['aversive', 'reward', 'baseline']
            
        for session in PARAMS['SESSIONS_LIST'][rat]:
            if channels == 'pool':
                sp = spectral_powers(precomputed_data = True, contexts_to_plot = contexts_to_plot, rats = [rat], 
                                     mode = 'all', 
                                     print_violin= False,
                                     baseline_normalize = norms['baseline_normalize'], 
                                     ME_normalize = norms['ME_normalize'], 
                                     sessions = [session])
            else:
                sp = spectral_powers(precomputed_data = True, contexts_to_plot = contexts_to_plot, rats = [rat],
                                     mode = channels, 
                                     print_violin= False,
                                     baseline_normalize = norms['baseline_normalize'], 
                                     ME_normalize = norms['ME_normalize'], 
                                     sessions = [session])
                
            for freq in freqs:
                channels_to_iter = ['v0', 'v1', 'v2', 'v3', 'v4'] if channels == 'pool' else channels[0]
                for chan in channels_to_iter:
                    for context in contexts:
                        corr, p_value = stats.pearsonr(sp[rat][context][f'{chan}_{freq}'], sp[rat][context][f'd_{freq}'])
                        corr_dict[freq][context].append(corr)
                        
                        if p_value < 0.05:
                            sig_corr_counter += 1
                            
                        all_corr_counter +=1
                    for context in ['morning', 'evening']:
                        sleep_mat = scipy.io.loadmat(f'{DATA_FOLDER}/Rat{rat}/Rat{rat}-{session}/Sleep_Scoring_with_wake.mat',simplify_cells=True)
                        chosen_context = fc.chooser(context, sleep_mat)
                        corr, p_value = stats.pearsonr(sp[rat][chosen_context][f'{chan}_{freq}'], sp[rat][chosen_context][f'd_{freq}'])
                        corr_dict[freq][context].append(corr)
                        
    print(f'significant correlations: {sig_corr_counter}/{all_corr_counter}')       
    return({'data': corr_dict, 'norms': norms, 'channel': channels})

def plot_correlation(corr_dict):
    
    norms = corr_dict['norms']
    channel = corr_dict['channel']
    with sns.plotting_context('notebook'):
        freqs = ['delta', 'theta', 'gamma', 'high gamma']
        pairs = [('aversive', 'reward')]
        fig, ax = plt.subplots(len(freqs), figsize=(10, 15), sharey = True)
        title = 'Correlation of spectral powers of REM epochs from all animals'
        subtitle = f'Ventral x dorsal correlation of spectral powers from channel {channel}\n baseline norm = {f'{norms['baseline_normalize']}'.upper()}       M/E norm = {f'{norms['ME_normalize']}'.upper()}'
      
        suptitle = plt.suptitle(title)
        suptitle.set(**SUPTITLE_SETTING)
        fig.text(0.5, 0.98, subtitle, va= 'top', ha='center', fontsize=14)
    
        sns.set_theme(style="darkgrid")
        
        plotting_data = corr_dict['data']
        
        for freq_ind in range(len(freqs)):
            freq = freqs[freq_ind]
            max_len = max(len(v) for v in plotting_data[freq].values())
            padded_data = {
                k: v + [np.nan]*(max_len - len(v))
                for k, v in plotting_data[freq].items()
            }
            df = pd.DataFrame(padded_data)
            df = df[['aversive', 'reward']]
            
            df_melted = df.melt(var_name='Group', value_name='Values')
            plotting_parameters = {'data' : df_melted, 
                                       'x': 'Group', 
                                       'y': 'Values', 
                                       'hue': 'Group', 
                                       'palette': 'Oranges',
                                      }
            sns.set_theme(style="darkgrid")
            sns.boxplot(**plotting_parameters, ax = ax[freq_ind])
            ax[freq_ind].set_title(freq, fontweight='bold')
            ax[freq_ind].set(xlabel = None)
            ax[freq_ind].set(ylabel = 'Pearson correlation values')
            
            annotator = Annotator(ax[freq_ind], pairs, **plotting_parameters)
            annotator.configure(test='Kruskal', 
                                loc='inside', 
                                text_format='star', 
                                show_test_name=True, 
                                line_height=0.01, 
                                fontsize=12, 
                                color='gray', 
                                verbose=False)           
            annotator.apply_and_annotate()
        plt.show()
        plt.savefig(os.path.join(SPECTRUM_PLOTS_PATH, 'power_correlation.png'))





### DATA LOADING ###

rat_datasets_all_channels = {}
for rat_N in RATS: 
    rat_datasets_all_channels[rat_N] = get_all_spectrum_plots_sessions(rat_N, _rat_dict = None, mode = 'all')

rat_datasets_merged_channels = {}
for rat_N in RATS: 
    rat_datasets_merged_channels[rat_N] = get_all_spectrum_plots_sessions(rat_N, _rat_dict = None, mode = 'merge_channels')




### FUNCTION CALLING ###

### figure 17 ###

print_all_spectrum_plots_rats(Hz = 100, mode = 'merge_channels', separate_rats = False) 

### figure 18 ###

spectral_powers(precomputed_data = True, contexts_to_plot = 'ARME', rats = RATS, mode = 'merge_channels', baseline_normalize = True, ME_normalize= False)

### figure 19 ###

spectral_powers(precomputed_data = True, contexts_to_plot = 'AR', rats = RATS, mode = 'merge_channels', baseline_normalize = True, ME_normalize= True)

### figure 20 ###

norms = {'baseline_normalize' : False,
                 'ME_normalize' : False
            }
data = prepare_for_power_correlation_plotting(norms, 'pool')
plot_correlation(data)


