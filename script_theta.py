
import yaml
import elephant
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import pandas as pd
from scipy.signal import hilbert
import numpy as np
import scipy
import os

import script_functions as fc


### PARAMETERS ###

SUPTITLE_SETTING = {'color': 'b',  'fontsize': 20, 'weight': 'bold'}

# DATA_FOLDER = '/CSNG/Morici_dataset/light_dataset_for_monika'
DATA_FOLDER = sys.argv[1:][0]

# SOURCE_FOLDER = '/CSNG/Morici_dataset/code'

with open(f'params_FM.yml') as f:
    PARAMS = yaml.safe_load(f)
RATS = PARAMS['RAT_LIST']
CONTEXTS = PARAMS['CONTEXTS']
DAY_PHASES = PARAMS['DAY_PHASES']

GREEN = '\033[92m'
RED = '\033[91m'
COLOR_END = '\033[0m'

HOME_PLOTS_FOLDER_NAME = 'plots'
THETA_PLOTS_FOLDER_NAME = 'theta_plots'

current_directory = os.getcwd()

HOME_PLOTS_PATH = os.path.join(current_directory, fr'{HOME_PLOTS_FOLDER_NAME}')
if not os.path.exists(HOME_PLOTS_PATH):
    os.makedirs(HOME_PLOTS_PATH)

THETA_PLOTS_PATH = os.path.join(HOME_PLOTS_PATH, fr'{THETA_PLOTS_FOLDER_NAME}')
if not os.path.exists(THETA_PLOTS_PATH):
    os.makedirs(THETA_PLOTS_PATH)


### DATA LOADING ###

animals = {}
for animal in RATS:
    rat_dict = fc.create_rat_dict(animal, [], DATA_FOLDER, has_dorsal=True)
    zscore_dict = fc.get_zscore_epoch_rat_dict(rat_dict)
    animals[animal] = zscore_dict



### DEFINITIONS ###

## helper functions

def merge_dicts_keep_keys(dict1, dict2):
    
    merged_dict = {}
    for context_name in dict1.keys():
        merged_dict[context_name] = {}
        for key in dict1[context_name].keys():
            merged_dict[context_name][key] = dict1[context_name][key] + dict2[context_name][key]

    return merged_dict

## fig 21

def zoom_on_freqs_lines(data, start, stop, freqs):
    freq_limits_dict = {'delta': {'min': 0,
                                  'max' : 4},
                        'theta': {'min': 4,
                                  'max' : 12},
                        'gamma': {'min': 30,
                                  'max' : 47},
                        'high gamma': {'min' : 53,
                                       'max' : 80}
                        }
    fig, ax = plt.subplots(5, sharey=True, figsize=(10, len(freqs)*9))
    sns.set_theme(style="darkgrid")
    fig.subplots_adjust(top=0.8, hspace=1)

    for line in range(5): 
        plot_data_buttered = {f : elephant.signal_processing.butter(data[f'v{str(line)}'][0][start:stop], 
                                                                 lowpass_frequency= freq_limits_dict[f]['max'], 
                                                                 highpass_frequency= freq_limits_dict[f]['min'], 
                                                                 sampling_frequency=1250) for f in freqs}
        plot_data = {f : scipy.stats.zscore(plot_data_buttered[f]) for f in plot_data_buttered.keys()}
        ax[line].set_title(f'v{line}')
        ax[line].set_title(f'Traces of theta, gamma, and gamma envelope in v{line} vHPC channel')
        ax[line].set_xlabel('Time [s]')
        ax[line].set_ylabel('Amplitude [a.u.]')
        gamma_hilbert = hilbert(plot_data['gamma'])
        amplitude_envelope = np.abs(gamma_hilbert)

        plot_data['gamma envelope'] = amplitude_envelope
        
        df = pd.DataFrame(plot_data)
        sns.lineplot(df, 
                     ax = ax[line],
                     palette = {'theta': 'orange', 'gamma': 'grey', 'gamma envelope': 'g'})
        lines = ax[line].get_lines()
        lines[0].set_linewidth(0.4)
    # plt.show()
    plt.savefig(f'{THETA_PLOTS_PATH}/freq_lines.png')


## fig 22

def zoom_on_freqs_heat(data, start, stop, freqs):
    freq_limits_dict = {'delta': {'min': 0,
                                  'max' : 4},
                        'theta': {'min': 4,
                                  'max' : 12},
                        'gamma': {'min': 30,
                                  'max' : 47},
                        'high gamma': {'min' : 53,
                                       'max' : 80}
                        }
    fig, ax = plt.subplots(len(freqs), sharey=True, figsize=(10, len(freqs)*3))
    fig.subplots_adjust(top=0.8, hspace=0.8)

    for freq_ind in range(len(freqs)): 
        freq = freqs[freq_ind]

        if freq == 'gamma envelope':
            plot_cmap_color = 'Oranges'
            
            plot_data_gamma = {f'v{i}' : elephant.signal_processing.butter(data[f'v{str(i)}'][0][start:stop], 
                                                                 lowpass_frequency= freq_limits_dict['gamma']['max'], 
                                                                 highpass_frequency= freq_limits_dict['gamma']['min'], 
                                                                 sampling_frequency=1250) for i in range(5)}
            plot_data = {f'v{i}' : np.abs(hilbert(plot_data_gamma[f'v{i}'])) for i in range(5)}
            
        else:
            plot_cmap_color = 'seismic'
            plot_data = {f'v{i}' : elephant.signal_processing.butter(data[f'v{str(i)}'][0][start:stop], 
                                                                     lowpass_frequency= freq_limits_dict[freq]['max'], 
                                                                     highpass_frequency= freq_limits_dict[freq]['min'], 
                                                                     sampling_frequency=1250) for i in range(5)}
          
            
        df = pd.DataFrame(plot_data)
       
        heatmap = sns.heatmap(df.T, 
                              cmap= plot_cmap_color,
                              center = 0,
                              ax = ax[freq_ind]
                              )
        ax[freq_ind].set_title(freq, fontweight='bold')
        ax[freq_ind].set_aspect(60)  # Aspect ratio of 2 (taller rows)
        ax[freq_ind].set_xlabel('Time [s]')
        ax[freq_ind].set_ylabel('Channel')
        xticks = np.arange(0, 1250, 100)  # show every 100th time point
        xtick_labels = np.arange(start, stop, 100)
        ax[freq_ind].set_xticks(xticks)
        ax[freq_ind].set_xticklabels(xtick_labels, rotation=45)
    # plt.show()
    plt.savefig(f'{THETA_PLOTS_PATH}/freq_heatmaps.png')


## fig 23

def prepare_data_for_PLV_violins_envelope(data, freqs, session, do_PLV = True, return_phase = False):

    first_freq = freqs[0]
    freq_limits_dict = {'delta': {'min': 0,
                                  'max' : 4},
                        'theta': {'min': 4,
                                  'max' : 12},
                        'gamma': {'min': 30,
                                  'max' : 47},
                        'high gamma': {'min' : 53,
                                       'max' : 80}
                        }

    return_PLV_data = {}
    return_data_for_corr = {}
    return_data_with_phase = {}
    
    for line in range(6): 
        data_index = 'd' if line == 5 else f'v{line}'
        plot_data_buttered = {f : [elephant.signal_processing.butter(data[data_index][i], 
                                                                 lowpass_frequency= freq_limits_dict[f]['max'], 
                                                                 highpass_frequency= freq_limits_dict[f]['min'], 
                                                                 sampling_frequency=1250) 
                              for i in range(len(data[data_index]))] 
                         for f in freqs} 

        hilbert_gamma = list(map(hilbert, plot_data_buttered[first_freq]))
        gamma_envelope = list(map(np.abs, hilbert_gamma))
        zscored_gamma_env = list(map(scipy.stats.zscore, gamma_envelope))
        plot_data_buttered['gamma_envelope'] = zscored_gamma_env
        
        hilbert_gamma_env = list(map(hilbert, plot_data_buttered['gamma_envelope']))
        gamma_env_phase = list(map(np.angle, hilbert_gamma_env))
        
        hilbert_theta = list(map(hilbert, plot_data_buttered['theta']))
        theta_phase = list(map(np.angle, hilbert_theta))

        return_data_for_corr[data_index] = {}
        return_data_for_corr[data_index]['gamma_envelope'] = gamma_envelope
        return_data_for_corr[data_index]['theta'] =   plot_data_buttered['theta']

        return_data_with_phase[data_index] = {}
        return_data_with_phase[data_index]['gamma_env_phase'] = gamma_env_phase
        return_data_with_phase[data_index]['theta_phase'] =  theta_phase

        plot_data = {'Theta x Gamma envelope' : list(map(elephant.phase_analysis.phase_locking_value, theta_phase, gamma_env_phase))}
        return_PLV_data[data_index] = plot_data
        
    if do_PLV:
        return return_PLV_data
    elif return_phase:
        return return_data_with_phase
    else:
        return return_data_for_corr

def get_shuffle_distribution(source_animals_data, channel, repetitions_num = 2000):

    dist_animal = '103'
    dist_session = list(source_animals_data[dist_animal].keys())[0]
    _dist_data = source_animals_data[dist_animal][dist_session]['vHPC']['aversive']['REM']
    _dist_data['d'] = source_animals_data[dist_animal][dist_session]['dHPC']['aversive']['REM']['d']
    data = prepare_data_for_PLV_violins_envelope(_dist_data, 
                                                 ['gamma', 'theta'], 
                                                 dist_session, 
                                                 do_PLV=False, 
                                                 return_phase = True)

    distribution = []
    epoch_num = 4
    
    for rep in range(0, repetitions_num // epoch_num):
        for epoch in range(0, epoch_num):
            base_phase_list = data[channel]['theta_phase'][epoch]
            shuffle_phase_list = data[channel]['gamma_env_phase'][epoch]
            
            shift = np.random.randint(0, len(shuffle_phase_list) - 1)
            shifted_phase_list = np.roll(shuffle_phase_list, shift) 
            distribution.append(elephant.phase_analysis.phase_locking_value(base_phase_list, shifted_phase_list))

    print(len(distribution))
    return distribution

def PLV_violins_envelope_kruskal(data, channel, title = '', ax = None, percentil = None, median = None):
    plot_data = {}

    # reduction of the level with one key
    for context in data.keys():
        plot_data[context] = data[context][channel]
  
    df = pd.DataFrame({k: pd.Series(v) for k, v in plot_data.items()})
    df_melted = df.melt(var_name='Group', value_name='Values')

    with sns.plotting_context('notebook'):
            plotting_parameters = {'data' : df_melted, 
                                   'x' : 'Group', 
                                   'y' : 'Values', 
                                   'hue' : 'Group', 
                                   'palette' : {'aversive': 'r', 'reward': 'y', 'morning': 'c', 'evening': 'm', 'baseline': 'g', 'shuffle': 'grey'},
                                   'linewidth' : 0.8,
                                   'ax': ax
                                  }
        
            sns.set_theme(style="darkgrid")
            sns.boxplot(**plotting_parameters)
            ax.set_title(f'channel {channel} {title}', fontweight='bold')
            ax.set_ylabel(None)
            ax.set_xlabel(None)
            ax.set_ylim([-0.01,0.2])

            if not percentil == None:
                plt.axhline(y=percentil, color='red', linestyle='--', linewidth=2,label = 'shuffle percentil 95') 
                plt.legend()
                    
            if not median == None:
                plt.axhline(y=median, color='blue', linestyle='--', linewidth=2,label = 'shuffle median') 
                plt.legend()
            
            pairs = [('baseline', 'aversive'), 
                     ('baseline', 'reward'), 
                     ('aversive', 'reward'), 
                    ]
        
            annotator = Annotator(pairs = pairs, **plotting_parameters)
            annotator.configure(test='Kruskal', 
                                loc='inside', 
                                text_format='star', 
                                show_test_name=True, 
                                line_height= 0.01, 
                                fontsize=12, 
                                color='gray', 
                                verbose=False)
                                    
            annotator.apply_and_annotate() 

def theta_gamma_envelope_PLV(freq_for_env, all_channels = False):

    plot_data = {}
    
    for context in CONTEXTS:
        one_context_plot_data = {}
        
        for animal in RATS:
            print(animal)
            
            for session in animals[animal].keys():
                freqs = [freq_for_env, 'theta']
                data = animals[animal][session]['vHPC'][context]['REM']
                data['d'] = animals[animal][session]['dHPC'][context]['REM']['d']
                prepared_data = prepare_data_for_PLV_violins_envelope(data, freqs, session)
                if len(one_context_plot_data.keys()) == 0:
                    one_context_plot_data = prepared_data
                else:
                    one_context_plot_data = merge_dicts_keep_keys(one_context_plot_data, prepared_data)
                    
        plot_data[context] = one_context_plot_data
        
    reduced_dict = {context: {} for context in plot_data.keys()}
    for context, con_d in plot_data.items():
            for chan, chan_d in con_d.items():
                reduced_dict[context][chan] = chan_d['Theta x Gamma envelope']
                
    # creating surrogate distribution
    distribution_v = get_shuffle_distribution(animals, 'v0', 2000)
    distribution_d = get_shuffle_distribution(animals, 'd', 2000)
    reduced_dict['shuffle'] = {chan: distribution_v for chan in ['v0', 'v1', 'v2', 'v3', 'v4']}
    reduced_dict['shuffle']['d'] = distribution_d

    title = f'PLV of theta and {freq_for_env} envelope, all animals'
    
    # plotting all channels
    if all_channels:
        
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(3, 2, figsize=(10, 10), sharey=True)
        ax = ax.flatten()
        suptitle = plt.suptitle(title)
        suptitle.set(**SUPTITLE_SETTING)
        suptitle.set(y = 0.91)
        fig.subplots_adjust(top=0.8, hspace=0.4)
        
        for index, chan in enumerate(['v0', 'v1', 'v2', 'v3', 'v4', 'd']):
            PLV_violins_envelope_kruskal(reduced_dict, chan, ax = ax[index])
        ax[0].set_ylabel('PLV')               
        ax[2].set_ylabel('PLV')               
        ax[4].set_ylabel('PLV')               
            
        plt.show()
    
    # plotting merged channels
    if not all_channels:
        merged_dict = {context: 
                    {'v': [], 'd': []} 
                    for context in reduced_dict.keys()}
        for context in reduced_dict.keys():
            for chan in ['v0', 'v1', 'v2', 'v3', 'v4']:
                merged_dict[context]['v'].extend(reduced_dict[context][chan])
            merged_dict[context]['d'] = reduced_dict[context]['d']
        
        fig, ax = plt.subplots(1, 2,  figsize=(8, 3), sharey=True)
        suptitle = plt.suptitle(title)
        suptitle.set(**SUPTITLE_SETTING)
        suptitle.set(y = 0.99)
        plt.tight_layout()
        fig.subplots_adjust(top=0.8, hspace=0.4)

        for index, chan in enumerate(['v', 'd']):

            PLV_violins_envelope_kruskal(merged_dict, chan, ax = ax[index])
            ax[index].set_ylabel('PLV')               
        # plt.show()
        plt.savefig(f'{THETA_PLOTS_PATH}/freq_heatmaps.png')




### FUNCTION CALLING ###

### figure 21 ###

zoom_on_freqs_lines(animals['103']['220715']['vHPC']['aversive']['REM'], 
                  6250, 7500, ['gamma', 'theta'])

### figure 22 ###

zoom_on_freqs_heat(animals['103']['220715']['vHPC']['aversive']['REM'], 
                  6250, 7500, ['gamma', 'theta', 'gamma envelope'])

### figure 23 ###

theta_gamma_envelope_PLV('gamma', False)

### figure 24 ###

theta_gamma_envelope_PLV('high gamma', False)

