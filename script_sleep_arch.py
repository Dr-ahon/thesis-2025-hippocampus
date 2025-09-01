import matplotlib.pyplot as plt
from scipy import stats
import scipy
import seaborn as sns
from statannotations.Annotator import Annotator
import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
import sys
import script_functions as fc



### PARAMETERS ###

# DATA_FOLDER = '/CSNG/Morici_dataset/light_dataset_for_monika'
DATA_FOLDER = sys.argv[1:][0]
# SOURCE_FOLDER = '/CSNG/Morici_dataset/code'
CONTEXTS = ['aversive', 'reward', 'baseline']
DAY_PHASES = ['morning', 'evening']
RATS = ['103', '127', '128', '132', '165']
SUPTITLE_SETTING = {'color': 'b', 'y': 1.02, 'fontsize': 20, 'weight': 'bold'}
GREEN = '\033[92m'
RED = '\033[91m'
COLOR_END = '\033[0m'

with open(f'params_FM.yml') as f:
    PARAMS = yaml.safe_load(f)

HOME_PLOTS_FOLDER_NAME = 'plots'
SLEEP_ARCH_PLOTS_FOLDER_NAME = 'sleep_arch_plots'

current_directory = os.getcwd()

HOME_PLOTS_PATH = os.path.join(current_directory, fr'{HOME_PLOTS_FOLDER_NAME}')
if not os.path.exists(HOME_PLOTS_PATH):
    os.makedirs(HOME_PLOTS_PATH)

SLEEP_ARCH_PLOTS_PATH = os.path.join(HOME_PLOTS_PATH, fr'{SLEEP_ARCH_PLOTS_FOLDER_NAME}')
if not os.path.exists(SLEEP_ARCH_PLOTS_PATH):
    os.makedirs(SLEEP_ARCH_PLOTS_PATH)


### DATA LOADING ###



### DEFINITIONS ###

## helper functions

def get_epoch_block_length_dict(rat_N, context_keyword, session = None):
    ### returns a dictionary of lengths of all blocks of sleeping epochs (REM, NREM, wake)

    if session == None:
        sessions = [x[7:] for x in os.listdir(f'{DATA_FOLDER}/Rat{rat_N}') ]
    else:
        sessions = [session]
    mint = (0, '')
    d = {session: {} for session in sessions}
   
    for session in sessions:

        # get intervals of epochs of different contexts
        sleep_mat = scipy.io.loadmat(f'{DATA_FOLDER}/Rat{rat_N}/Rat{rat_N}-{session}/Sleep_Scoring_with_wake.mat',simplify_cells=True)
        context = fc.chooser(context_keyword, sleep_mat)
        nrem = [('NREM', list(interval)) for interval in sleep_mat['NREM'][context]]
        rem = [('REM', list(interval)) for interval in sleep_mat['REM'][context]]
        wake = [('wake', list(interval)) for interval in sleep_mat['wake'][context]]

        # return dict inicialisation
        d['context_keyword'] = context_keyword
        d[session]['NREM'] = nrem
        d[session]['REM'] = rem
        d[session]['wake'] = wake
        # session with all the phases chronologically
        d[session]['sorted_session'] = sorted(nrem + rem + wake, key = lambda x: x[1][0] )
        # tuples of (epoch phase, epoch length)
        d[session]['data'] = list(map(lambda x: (x[0], sec_to_min(x[1][1] - x[1][0])), d[session]['sorted_session']))
        d[session]['REM_length'] = list(map(lambda x: (x[0], sec_to_min(x[1][1] - x[1][0])), d[session]['REM']))
        d[session]['NREM_length'] = list(map(lambda x: (x[0], sec_to_min(x[1][1] - x[1][0])), d[session]['NREM']))
       
        if (float(d[session]['data'][0][1]) > mint[0] ) : mint = (d[session]['data'][0][1], session)

    return d

def sec_to_min(x):
    return x / 60

def get_total_duration(epochs, session_sleep_mat, context):
    
    concat_list = [x 
                   for list in [session_sleep_mat[epoch][context] for epoch in epochs]
                   for x in list]
    return sum([a[1] - a[0] for a in concat_list])

def normalize(data_dict, rat, sleep_mat, keys, ME_norm_operation, sleep_length_norm = False):
   
    data = {context: 
            {key: []
             for key in keys} 
            for context in CONTEXTS + DAY_PHASES}    
    
                
    for key in keys:
        morning_normalizing_factor = np.mean(fc.flatten([data_dict['morning'][key][rat][session] 
                                                         for session 
                                                         in data_dict['morning'][key][rat].keys()]))
        evening_normalizing_factor = np.mean(fc.flatten([data_dict['evening'][key][rat][session] 
                                                         for session 
                                                         in data_dict['evening'][key][rat].keys()]))
                
        for context in CONTEXTS + DAY_PHASES:
            d = get_epoch_block_length_dict(rat, context)
            
            for session in data_dict[context][key][rat].keys():
                sleep_length_normalizing_factor = sum([x[1] for x in  d[session]['data']])
                
                if ME_norm_operation == 'div':
                   
                    if fc.chooser('morning', sleep_mat) == context or context == 'morning':
                        arr = np.array(data_dict[context][key][rat][session]) / morning_normalizing_factor
                    elif fc.chooser('evening', sleep_mat) == context or context == 'evening':
                        arr = np.array(data_dict[context][key][rat][session]) / evening_normalizing_factor
                elif ME_norm_operation == 'minus':
                   
                    if fc.chooser('morning', sleep_mat) == context or context == 'morning':
                        arr = np.array(data_dict[context][key][rat][session]) - morning_normalizing_factor
                    elif fc.chooser('evening', sleep_mat) == context or context == 'evening':
                        arr = np.array(data_dict[context][key][rat][session]) - evening_normalizing_factor
                else:
                    arr = data_dict[context][key][rat][session] 

                if sleep_length_norm:
                    data[context][key].extend(arr / sleep_length_normalizing_factor)
                else:
                    data[context][key].extend(arr)
                                                      
    return data

## fig 3

def get_any_plots_sleep_architecture_by_context(args):
    
    colors = {'NREM': 'mediumblue',
              'REM': 'indianred',
              'wake': '#fff75e'}
    added_to_legend = {'mediumblue': False,
                       '#fff75e': False,
                       'indianred': False}
    height_per_session = 0.5
    
    fig_height = max(3, len(args) * height_per_session)

    with sns.plotting_context('notebook'):
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(10, fig_height))
        
        for i in range(len(args)):
            
            name, arg = args[i]
            rat_N, context, _session = arg
            d = get_epoch_block_length_dict(rat_N, context, session = _session)
            epochs = d[_session]['data'].copy()
            
            pos = 0
            for (epoch, length) in epochs:
                bar_color = colors[epoch]
                plt.barh(
                    y = name, 
                    width = [length], 
                    height = height_per_session,
                    color = bar_color, 
                    left = pos,
                    linewidth = 0,
                    label = epoch if not added_to_legend[bar_color] else None)
                pos+=length
                added_to_legend[bar_color] = True
    
        plt.xlabel('Time [min]')
        plt.ylabel('Sessions')
        plt.yticks(np.arange(len([name for (name, a) in args])), [name.replace('_', ' ') for (name, a) in args])
        k = d['context_keyword']
        plt.legend(bbox_to_anchor=(1.0, 0.55))
        plt.show()
        plt.savefig(f'{SLEEP_ARCH_PLOTS_PATH}/sleep_arch.png')

## fig 4, 5, 6

def total_phases_boxplots(test_name, ME_norm_operation = None, sleep_length_norm = False):
    ## generates figure with total REM, total NREM and total sleep
    ## one datapoint = sum of REM length in one session

    measures = ['total_REM', 'total_NREM', 'total_sleep']
    measure_names = ['Total length of REM periods', 'Total length of NREM periods', 'Total sleep length (REM + NREM)']
    
    data = {context: 
            {measure: 
             [] 
             for measure in measures} 
            for context in CONTEXTS + DAY_PHASES}
    
    for rat in RATS:
        
        rat_dict = {context: 
                    {measure: 
                     {rat: {}
                      for rat in RATS}
                     for measure in measures} 
                    for context in CONTEXTS + DAY_PHASES}
        
        for context in CONTEXTS + DAY_PHASES:
            sessions = [x[7:] for x in os.listdir(f'{DATA_FOLDER}/Rat{rat}')]
            for _session in sessions:
                session_sleep_mat = scipy.io.loadmat(f'{DATA_FOLDER}/Rat{rat}/Rat{rat}-{_session}/Sleep_Scoring_with_wake.mat',simplify_cells=True)
                context_key = fc.chooser(context, session_sleep_mat)
                
                rat_dict[context]['total_REM'][rat][_session] = [sec_to_min(get_total_duration(['REM'], session_sleep_mat, context_key))]
                rat_dict[context]['total_NREM'][rat][_session] = [sec_to_min(get_total_duration(['NREM'], session_sleep_mat, context_key))]
                rat_dict[context]['total_sleep'][rat][_session] = [sec_to_min(get_total_duration(['NREM', 'REM'], session_sleep_mat, context_key))]
        
        normalized_dict = normalize(rat_dict.copy(), rat, session_sleep_mat, measures, ME_norm_operation, sleep_length_norm)
        
        for context in CONTEXTS + DAY_PHASES :
            for measure in measures:
                data[context][measure].extend(normalized_dict[context][measure])
    
    fig, ax = plt.subplots(1, len(measures), figsize = (15,5))
    y_tick_size = 12
    
    if ME_norm_operation != None:
        data = {'aversive': data['aversive'], 'reward': data['reward']}
        pairs = [('aversive', 'reward')]  
        suptitle = plt.suptitle(f'M/E {ME_norm_operation} norm')
        y_label = '[norm. u.]'
        x_tick_size = 15

        if sleep_length_norm:
            suptitle = plt.suptitle(f'M/E {ME_norm_operation} norm, sleep session length norm')
            
    else:
        pairs = [('baseline', 'morning'), ('baseline', 'evening'), ('baseline', 'aversive'), ('baseline', 'reward'), ('morning', 'evening'), ('aversive', 'reward')]
        y_label = 'Time [min]'
        suptitle = plt.suptitle(f'Pre-normalization')
        x_tick_size = 10
        
    suptitle.set(**SUPTITLE_SETTING)
    
                
    for i in range(len(measures)):
        measure = measures[i]
        df = pd.DataFrame({key: data[key][measure] for key in data.keys()})
        df_melted = df.melt(var_name='Group', value_name='Values')

        with sns.plotting_context('notebook'):
            sns.set_theme(style="darkgrid", )
            plotting_parameters = {'data' : df_melted, 
                                   'x' : 'Group', 
                                   'y' : 'Values', 
                                   'hue' : 'Group', 
                                   'palette' : {'aversive': 'r', 'reward': 'y', 'morning': 'c', 'evening': 'm', 'baseline': 'g'}
                                  }
            sns.boxplot(**plotting_parameters, ax = ax[i])
            title = measure_names[i]
            ax[i].set_title(title, fontweight='bold')
            ax[i].set_ylabel(None)
            ax[i].set_xlabel(None)
            ax[i].xaxis.set_tick_params(labelsize = x_tick_size)
            ax[i].yaxis.set_tick_params(labelsize = y_tick_size)
            annotator = Annotator(ax[i], pairs, **plotting_parameters)
            annotator.configure(test=test_name, 
                                loc='inside', 
                                text_format='star', 
                                show_test_name=True, 
                                line_height=0.01, 
                                fontsize=12, 
                                color='gray', 
                                verbose=False)           
            annotator.apply_and_annotate()
    ax[0].set_ylabel(y_label)
    # plt.show()
    plt.savefig(f'{SLEEP_ARCH_PLOTS_PATH}/total_phases_MEnorm_{ME_norm_operation}_sleep_length_norm_{sleep_length_norm}.png')

                
## fig 7, 8, 9

def violinplots(test_name, axes, ME_norm_operation = None, sleep_length_norm = False, data_folder = DATA_FOLDER):
    phases =  ['REM', 'NREM', 'wake']
    data = {context: 
            {phase: []
             for phase in phases} 
            for context in CONTEXTS + DAY_PHASES}
    
    for rat in RATS:
       
        rat_dict = {context: 
                    {phase: 
                      {rat: {}
                      for rat in RATS}
                     for phase in phases} 
                    for context in CONTEXTS + DAY_PHASES}
                
        for context in CONTEXTS + DAY_PHASES:
            sessions = [x[7:] for x in os.listdir(f'{data_folder}/Rat{rat}')]
            for _session in sessions:
                session_sleep_mat = scipy.io.loadmat(f'{data_folder}/Rat{rat}/Rat{rat}-{_session}/Sleep_Scoring_with_wake.mat',simplify_cells=True)
                context_key = fc.chooser(context, session_sleep_mat)
                for phase in phases:
                    rat_dict[context][phase][rat][_session] = ([sec_to_min(a[1] - a[0]) for a in session_sleep_mat[phase][context_key]])
                
        normalized_dict = normalize(rat_dict.copy(), rat, session_sleep_mat, phases, ME_norm_operation, sleep_length_norm)
        for context in  CONTEXTS + DAY_PHASES:
            for phase in phases:            
                data[context][phase].extend(normalized_dict[context][phase])
    
    y_tick_size = 12
    
    if ME_norm_operation != None:
        data = {'aversive': data['aversive'], 'reward': data['reward']}
          
    for i in range(len(phases)):
        phase = phases[i]
        max_len = max([len(data[key][phase]) for key in  data.keys()])

        # addition of NaNs for length evening
        for key in data.keys():
            data[key][phase] = [*data[key][phase], *[np.nan] * (max_len-len(data[key][phase]))]
               
            
        df = pd.DataFrame({key: data[key][phase] for key in data.keys()})
        df_melted = df.melt(var_name='Group', value_name='Values')
        if ME_norm_operation == None:
            pairs = [('baseline', 'morning'), ('baseline', 'evening'), ('baseline', 'aversive'), ('baseline', 'reward'), ('morning', 'evening'), ('aversive', 'reward')]
            x_tick_size = 10
            
        else:
            pairs = [('aversive', 'reward')]  
            x_tick_size = 15
            
        plots = []
        with sns.plotting_context('notebook'):
            plotting_parameters = {'data' : df_melted, 
                                   'x' : 'Group', 
                                   'y' : 'Values', 
                                   'hue' : 'Group', 
                                    'palette' : {'aversive': 'r', 'reward': 'y', 'morning': 'c', 'evening': 'm', 'baseline': 'g'},
                                   'linewidth' : 0.8,
                                   'inner' : 'box',
                                   'edgecolor' : (0, 0, 0, 0.2),
                                   'cut' : 0
                                  }
            sns.set_theme(style="darkgrid")
            violinplot = sns.violinplot(**plotting_parameters, ax = axes[i])
            title=f'Length of {phase} epochs'
            axes[i].set_title(title, fontweight='bold')
            axes[i].set_ylabel(None)
            axes[i].set_xlabel(None)
            axes[i].axhline(0, color='r', linewidth=0.4)
            axes[i].xaxis.set_tick_params(labelsize = x_tick_size)
            axes[i].yaxis.set_tick_params(labelsize = y_tick_size)
            
            annotator = Annotator(axes[i], pairs, **plotting_parameters)
            annotator.configure(test=test_name, 
                                loc='inside', 
                                text_format='star', 
                                show_test_name=True, 
                                line_height=0.01, 
                                fontsize=12, 
                                color='gray', 
                                verbose=False)           
            annotator.apply_and_annotate()
            plots.append((violinplot, annotator))
    return plots

def nums_boxplots(test_name, ME_norm_operation = None,  sleep_length_norm = False, print_plot = True, data_folder = DATA_FOLDER):
    ## total REM, total NREM and total sleep

    measures = ['REM_epoch_N', 'NREM_epoch_N', 'wake_epoch_N']
    measure_names = ['Number of REM epochs', 'Number of NREM epochs', 'number of wake epochs']
    
    data = {context: 
            {measure: 
             [] 
             for measure in measures} 
            for context in CONTEXTS + DAY_PHASES}
    
    for rat in RATS:
        counter = 0
        
        rat_dict = {context: 
                    {measure: 
                     {rat: {}
                      for rat in RATS}
                     for measure in measures} 
                    for context in CONTEXTS + DAY_PHASES}
        for context in CONTEXTS + DAY_PHASES:
            sessions = [x[7:] for x in os.listdir(f'{data_folder}/Rat{rat}')]
            for _session in sessions:
                session_sleep_mat = scipy.io.loadmat(f'{data_folder}/Rat{rat}/Rat{rat}-{_session}/Sleep_Scoring_with_wake.mat',simplify_cells=True)
                context_key = fc.chooser(context, session_sleep_mat)
                rat_dict[context]['REM_epoch_N'][rat][_session] = [len(session_sleep_mat['REM'][context_key])]
                rat_dict[context]['NREM_epoch_N'][rat][_session] = [len(session_sleep_mat['NREM'][context_key])]
                if (context == 'aversive' or context == 'reward'):
                    counter += sum(rat_dict[context]['REM_epoch_N'][rat][_session])
                rat_dict[context]['wake_epoch_N'][rat][_session] = [len(session_sleep_mat['wake'][context_key])]
        
        normalized_dict = normalize(rat_dict.copy(), rat, session_sleep_mat, measures, ME_norm_operation, sleep_length_norm)
        
        for context in  CONTEXTS + DAY_PHASES:
            for measure in measures:
                data[context][measure].extend(normalized_dict[context][measure])
        
    if not print_plot:
        return data
    
    viol_plots_num = len(measures)
    fig, ax = plt.subplots(2, viol_plots_num, figsize = (15, 10))
    violinplots('Kruskal', [ax[0,0], ax[0,1], ax[0,2]], ME_norm_operation = ME_norm_operation, sleep_length_norm = sleep_length_norm, data_folder = data_folder)
    y_tick_size = 12
    
    if ME_norm_operation != None:
        data = {'aversive': data['aversive'], 'reward': data['reward']}
        pairs = [('aversive', 'reward')] 
        suptitle = plt.suptitle(f'M/E {ME_norm_operation} norm')    
        y_label = '[norm. u.]'
        x_tick_size = 15
        
        if sleep_length_norm:
            suptitle = plt.suptitle(f'M/E {ME_norm_operation} norm, sleep session length norm')
       
    else:
        pairs = [('baseline', 'morning'), 
                 ('baseline', 'evening'), 
                 ('baseline', 'aversive'), 
                 ('baseline', 'reward'), 
                 ('morning', 'evening'), 
                 ('aversive', 'reward')]
        y_label = 'Time [min]'
        suptitle = plt.suptitle(f'Pre-normalization')
        x_tick_size = 10
        
        
    suptitle.set(**SUPTITLE_SETTING)
    suptitle.set(y = 0.97)

    for i in range(len(measures)):
        measure = measures[i]
        df = pd.DataFrame({key: data[key][measure] for key in data.keys()})
        df_melted = df.melt(var_name='Group', value_name='Values')
        with sns.plotting_context('notebook'):
            sns.set_theme(style="darkgrid")
            plotting_parameters = {'data' : df_melted, 
                                   'x' : 'Group', 
                                   'y' : 'Values', 
                                   'hue' : 'Group', 
                                    'palette' : {'aversive': 'r', 'reward': 'y', 'morning': 'c', 'evening': 'm', 'baseline': 'g'}
                                  }
            sns.boxplot(**plotting_parameters, ax = ax[1, i])
            title = f'{measure_names[i]}'
            ax[1,i].set_title(title, fontweight='bold')
            ax[1,i].set_xlabel(None)
            ax[1,i].set_ylabel(None)
            ax[1,i].axhline(0, color='r', linewidth=0.4)
            ax[1,i].xaxis.set_tick_params(labelsize = x_tick_size)
            ax[1,i].yaxis.set_tick_params(labelsize = y_tick_size)
            if ME_norm_operation != None:
                ax[1, 0].set_ylabel(y_label)
            elif sleep_length_norm == False:
                ax[1, 0].set_ylabel('Number of epochs')
                
            annotator = Annotator(ax[1, i], pairs, **plotting_parameters)
            annotator.configure(test=test_name, 
                                loc='inside', 
                                text_format='star', 
                                show_test_name=True, 
                                line_height=0.01, 
                                fontsize=12, 
                                color='gray', 
                                verbose=False)           
            annotator.apply_and_annotate()
        
    ax[0,0].set_ylabel(y_label)
    plt.savefig(f'{SLEEP_ARCH_PLOTS_PATH}/epoch_length_MEnorm_{ME_norm_operation}_sleep_length_norm_{sleep_length_norm}.png')

    plt.show()
                
## fig 10, 11, 12

def first_NREM_boxplots(test_name, ME_norm_operation = None, sleep_length_norm = False):
    ## duration of the first wake, REM/NREM length, sleep/wake length
    
    measures = ['first_NREM', 'REM_to_NREM_length', 'sleep_to_wake_length']
    measure_names = ['Duration of the first wake', 'The ratio of REM to NREM', 'The ratio of REM + NREM to wake']

    data = {context: 
            {measure: 
             [] 
             for measure in measures} 
            for context in CONTEXTS + DAY_PHASES}
    ratios = []
    for rat in RATS:
       
        rat_dict = {context:
                    {measure:
                     {rat: {}
                      for rat in RATS}
                     for measure in measures}
                    for context in CONTEXTS + DAY_PHASES}
        
        for context in CONTEXTS + DAY_PHASES:
            sessions = [x[7:] for x in os.listdir(f'{DATA_FOLDER}/Rat{rat}')]
            for _session in sessions:
                session_sleep_mat = scipy.io.loadmat(f'{DATA_FOLDER}/Rat{rat}/Rat{rat}-{_session}/Sleep_Scoring_with_wake.mat',simplify_cells=True)
                session_structure_mat = scipy.io.loadmat(f'{DATA_FOLDER}/Rat{rat}/Rat{rat}-{_session}/Session_Structure.mat',simplify_cells=True)
                context_key = fc.chooser(context, session_sleep_mat)
                rat_dict[context]['first_NREM'][rat][_session] = [sec_to_min(session_sleep_mat['NREM'][context_key][0][0] - session_structure_mat['TimeStamps']['Sleep'][context_key.capitalize()][0])]
                rat_dict[context]['REM_to_NREM_length'][rat][_session] = [(get_total_duration(['REM'], session_sleep_mat, context_key) / get_total_duration(['NREM'], session_sleep_mat, context_key))]
                ratios.append([rat_dict[context]['REM_to_NREM_length'][rat][_session], rat, _session, context])
                rat_dict[context]['sleep_to_wake_length'][rat][_session] = [(get_total_duration(['REM', 'NREM'],  session_sleep_mat, context_key) / get_total_duration(['wake'], session_sleep_mat, context_key))]

        normalized_dict = normalize(rat_dict.copy(), rat, session_sleep_mat, measures, ME_norm_operation, sleep_length_norm)

        for context in  CONTEXTS + DAY_PHASES:
            for measure in measures:
                data[context][measure].extend(normalized_dict[context][measure])
    
    fig, ax = plt.subplots(1, len(measures),  figsize = (17,5))

    pairs = [('baseline', 'morning'), ('baseline', 'evening'), ('baseline', 'aversive'), ('baseline', 'reward'), ('morning', 'evening'), ('aversive', 'reward')]
    y_label = 'Time [min]'
    y_tick_size = 12
    
    if ME_norm_operation != None:
        data = {'aversive': data['aversive'], 'reward': data['reward']}
        pairs = [('aversive', 'reward')]  
        y_label = '[norm. u.]'
        x_tick_size = 15
        
        if sleep_length_norm:
            title = plt.suptitle(f'M/E {ME_norm_operation} norm, sleep session length norm')
        else:
            title = plt.suptitle(f'M/E {ME_norm_operation} norm')
            
    else:
        title = plt.suptitle(f'Pre-normalization')
        x_tick_size = 10
        
    title.set(**SUPTITLE_SETTING)
    
    for i in range(len(measures)):
        measure = measures[i]

        df = pd.DataFrame({key: data[key][measure] for key in data.keys()})
        df_melted = df.melt(var_name='Group', value_name='Values')
            
        with sns.plotting_context('notebook'):
            sns.set_theme(style="darkgrid")
            plotting_parameters = {'data' : df_melted, 
                                   'x' : 'Group', 
                                   'y' : 'Values', 
                                   'hue' : 'Group', 
                                   'palette' : {'aversive': 'r', 'reward': 'y', 'morning': 'c', 'evening': 'm', 'baseline': 'g'}
                                  }
            sns.boxplot(**plotting_parameters, ax = ax[i])
            title = measure_names[i]
            ax[i].set_title(title, fontweight='bold')
            annotator = Annotator(ax[i], pairs, **plotting_parameters)
            annotator.configure(test=test_name, 
                                loc='inside', 
                                text_format='star', 
                                show_test_name=True, 
                                line_height=0.01, 
                                fontsize=12,
                                color='gray', 
                                verbose=False)
            annotator.apply_and_annotate()
            ax[i].set_ylabel(None)
            ax[i].set_xlabel(None)
            ax[i].xaxis.set_tick_params(labelsize = x_tick_size)
            ax[i].yaxis.set_tick_params(labelsize = y_tick_size)
        
    ax[0].set_ylabel(y_label)
    # plt.show()
    plt.savefig(f'{SLEEP_ARCH_PLOTS_PATH}/first_wake_and_ratios_MEnorm_{ME_norm_operation}_sleep_length_norm_{sleep_length_norm}.png')

                
## fig 13

def get_scatter_plots(rat_N, test_name = 'Kruskal', reg = False, only_fig=False):
    
    sessions = [x[7:] for x in os.listdir(f'{DATA_FOLDER}/Rat{rat_N}') ]
    
    if only_fig:
        context = 'baseline'
        phase = 'NREM'
        d = get_epoch_block_length_dict(rat_N, context)
        plt.figure(figsize=(6, 2))
        plt.xlabel("Epoch index")
        plt.title(f"Linear regression fits of {context} {phase} sleep sessions of rat {rat_N}")  
        for session in sessions:
            data = {'Epoch length':[x[1] for x in d[session][f'{phase}_length']]}
            coef = np.polyfit(range(len(data['Epoch length'])), data['Epoch length'], 1)
            slope = coef[0]
            df = pd.DataFrame(data)
            df['Epoch index'] = df.index
            
            with sns.plotting_context('notebook'):
                plotting_parameters = {'data' : df, 
                                       'x' : 'Epoch index', 
                                       'y' : 'Epoch length', 
                                       'label' : session
                                      }
                sns.set_theme(style="darkgrid")
                sns.regplot(**plotting_parameters)
        plt.legend(title = 'sessions', bbox_to_anchor=(1.05, 1.03))
        plt.ylabel('Time [min]')
        plt.savefig(f'{SLEEP_ARCH_PLOTS_PATH}/epoch_length_slopes.png', bbox_inches='tight')

        return
    
    fig, ax = plt.subplots(5, 2, sharey=True, tight_layout = False)
    fig.set_figwidth(12)
    fig.set_figheight(18)
    
    if reg:
        suptitle = plt.suptitle(f'Linear regression of lengths of epochs of distinct sleep phase \n across all sessions of rat {rat_N}')
        y_label = 'Slope'
    else:
        suptitle = plt.suptitle(f'Lengths of epochs of distinct sleep phase across all sessions of rat {rat_N}')
        y_label = 'Time [min]'
        
    suptitle.set(**SUPTITLE_SETTING)
    suptitle.set(y = 1)
    sessions = [x[7:] for x in os.listdir(f'{DATA_FOLDER}/Rat{rat_N}') ]
    phases = ['REM', 'NREM']
    contexts = ['morning', 'evening', 'baseline', 'aversive', 'reward']

    box_data = {phase: {context: [] for context in contexts} for phase in phases}

    for phase_index in range(len(phases)):
        phase = phases[phase_index]
        for context_index in range(len(contexts)):
            context = contexts[context_index]
            d = get_epoch_block_length_dict(rat_N, context)
            for session in sessions:
                
                data = {'values':[x[1] for x in d[session][f'{phase}_length']]}
                coef = np.polyfit(range(len(data['values'])), data['values'], 1)
                slope = coef[0]
                box_data[phase][context].append(slope)
                df = pd.DataFrame(data)

                with sns.plotting_context('notebook'):
                    plotting_parameters = {'data' : df, 
                                           'x' : df.index, 
                                           'y' : 'values', 
                                           'label' : session
                                          }
                    sns.set_theme(style="darkgrid")
                    if reg:
                        sns.regplot(**plotting_parameters, ax = ax[ context_index, phase_index])
                    else:
                        sns.lineplot(**plotting_parameters, ax = ax[ context_index, phase_index])
                        
                    title=f'{context} {phase}'
                    ax[context_index, phase_index].set_title(title, fontweight='bold')
                    ax[context_index, 0].set_ylabel(y_label)
                    ax[context_index, 1].set_ylabel(None)
                    ax[context_index, phase_index].set_xlabel(None)
                    ax[context_index, phase_index].legend(title = 'sessions')
    
    ax[len(contexts) - 1, 0].set_xlabel('index')
    ax[len(contexts) - 1, 1].set_xlabel('index')

    fig_box, ax_box = plt.subplots(1, len(phases), sharey=True)
    fig_box.set_figwidth(16)
    fig_box.set_figheight(10)

    suptitle = plt.suptitle(f'Slopes of linear fits of all sessions of rat {rat_N}')
    suptitle.set(**SUPTITLE_SETTING)

    for i in range(len(phases)):
        box_df = pd.DataFrame(box_data[phases[i]])
        box_df_melted = box_df.melt(var_name='Group', value_name='Values')
        col_palette = ['g', 'g', 'm', 'b', 'b']
        pairs = [('baseline', 'morning'), ('baseline', 'evening'), ('baseline', 'aversive'), ('baseline', 'reward'), ('morning', 'evening'), ('aversive', 'reward')]
        plt.setp(ax_box[0], ylim=(-1, 1))
        with sns.plotting_context('notebook'):
            sns.set_theme(style="darkgrid", )
            plotting_parameters = {'data' : box_df_melted, 
                                   'x' : 'Group', 
                                   'y' : 'Values', 
                                   'hue' : 'Group', 
                                   'palette' : col_palette
                                  }
            sns.boxplot(**plotting_parameters, ax = ax_box[i])
            title = phases[i]
            ax_box[i].set_title(title, fontweight='bold')
            ax_box[i].set_xlabel(None)
            ax_box[i].set_ylabel('Slopes')
            annotator = Annotator(ax_box[i], pairs, **plotting_parameters)
            annotator.configure(test=test_name, 
                                loc='inside', 
                                text_format='star', 
                                show_test_name=True, 
                                line_height=0.01, 
                                fontsize=12, 
                                color='gray', 
                                verbose=False)           
            annotator.apply_and_annotate()
    plt.show()         
    
## fig 14, 15, 16

def get_slope_boxplots( test_name = 'Kruskal', slope_dist = False, scatter = False, zero_test = False):
    
    # data preparation
    phases = ['REM', 'NREM']
    contexts = [ 'aversive', 'reward', 'baseline', 'morning', 'evening']
    scatter_data = {phase:
                    {context:
                     {rat_N: []
                      for rat_N in RATS}
                     for context in contexts}
                    for phase in phases}
    box_data = {phase:
                {context: []
                 for context in contexts}
                for phase in phases} 

    for rat_N in RATS:
        sessions = [x[7:] for x in os.listdir(f'{DATA_FOLDER}/Rat{rat_N}') ]
        
        for phase_index in range(len(phases)):
            phase = phases[phase_index]
            
            for context_index in range(len(contexts)):
                context = contexts[context_index]
                d = get_epoch_block_length_dict(rat_N, context)
                
                for session in sessions:
                    
                    data = {'values':[x[1] for x in d[session][f'{phase}_length']]}
                    coef = np.polyfit(range(len(data['values'])), data['values'], 1)
                    slope = coef[0]
                    scatter_data[phase][context][rat_N].append(slope)
                    box_data[phase][context].append(slope)

###############

    # Wilcoxon zero test
    if zero_test:
        
        w_test_file_path = os.path.join(SLEEP_ARCH_PLOTS_PATH, "wilcoxon_zero_test.txt")
        if not os.path.isfile(w_test_file_path):
            open(w_test_file_path, "x") 
        

        with open(w_test_file_path, "w") as f:
            f.write('wilcoxon statistics against zero')
            f.write('\n\n')
            pairs = [('aversive', 'zero'), ('reward', 'zero'), ('aversive', 'reward')]
            
            for phase_index in range(len(phases)):
                phase = phases[phase_index]
                f.write(phase)
                f.write('\n')

                test_data = pd.DataFrame(box_data[phase])
                test_data['zero'] = [0 for i in range(len(test_data['morning']))]
                for (x, y) in pairs:
                    f.write(f'\t testing {x} vs {y}:\n')
                    statistic, p_value = stats.wilcoxon(test_data[x], test_data[y])
                    f.write(f"\t\t P-value: {p_value}\n")
                    if p_value < 0.05:
                        f.write(f"\t\t {GREEN}Reject null hypothesis: {COLOR_END} 'The two samples are significantly different.\n")
                    else:
                        f.write(f" \t\t {RED}Fail to reject null hypothesis: {COLOR_END} The two samples are not significantly different.\n")
            f.write('\n')

#################
                    
    if scatter == True:
        fig_scatter, ax_scatter = plt.subplots(2, 5, sharey='row')
        fig_scatter.set_figwidth(13)
        fig_scatter.set_figheight(10)

        title = plt.suptitle('Slopes of epoch lengths for each animal and context')
        title.set(**SUPTITLE_SETTING)
        for phase_index in range(len(phases)):
            phase = phases[phase_index]
            for context_index in range(len(contexts)):
                context = contexts[context_index]
                
                plotting_data = [{'value': value, 'rat_N': rat_N}
                                for rat_N, values in scatter_data[phase][context].items()
                                for value in values] 
               
                scatter_df = pd.DataFrame(plotting_data)
                with sns.plotting_context('notebook'):
                    plotting_parameters = {'data' : scatter_df, 
                                           'x' : 'rat_N', 
                                           'y' : 'value', 
                                           'hue': 'rat_N',
                                           'legend': False,
                                           'palette': "deep"
                                          }
                    sns.set_theme(style="darkgrid")
                    sns.scatterplot(**plotting_parameters, ax = ax_scatter[phase_index, context_index])
                    ax_scatter[phase_index, context_index].set_title(f'{context} {phase}', fontweight='bold')
                    ax_scatter[phase_index, context_index].set_ylabel('Slope')
                    ax_scatter[phase_index, context_index].set_xlabel(None)
                    ax_scatter[phase_index, context_index].axhline(0, color='red', linewidth=0.4)
                    ax_scatter[phase_index, 2].set_xlabel('Animal', fontsize=13, labelpad = 10.0)
        plt.subplots_adjust(hspace=0.5, wspace=0.1)
    # plt.show()
    plt.savefig(f'{SLEEP_ARCH_PLOTS_PATH}/slope_scatter_{test_name}.png')


######################

    if slope_dist:
        fig_box, ax_box = plt.subplots(1, 2, sharey='row')
        fig_box.set_figwidth(13)
        fig_box.set_figheight(7)
    
        title = plt.suptitle(f'Slope distribution across all animals \n ({test_name})')
        title.set(**SUPTITLE_SETTING)
        
        for phase_index in range(len(phases)):
            phase = phases[phase_index]
            box_df = pd.DataFrame(box_data[phase])
            box_df_melted = box_df.melt(var_name='Group', value_name='Values')
            pairs = [('baseline', 'morning'), ('baseline', 'evening'), ('baseline', 'aversive'), ('baseline', 'reward'), ('morning', 'evening'), ('aversive', 'reward')]
            
            with sns.plotting_context('notebook'):
                sns.set_theme(style="darkgrid", )
                plotting_parameters = {'data' : box_df_melted, 
                                       'x' : 'Group', 
                                       'y' : 'Values', 
                                       'hue' : 'Group', 
                                       'palette' : {'aversive': 'r', 'reward': 'y', 'morning': 'c', 'evening': 'm', 'baseline': 'g'}
                                       # 'palette' : col_palette
                                      }
                                        
                sns.boxplot(**plotting_parameters, ax = ax_box[phase_index])
                title = phases[phase_index]
                ax_box[phase_index].set_title(title, fontweight='bold')
                ax_box[phase_index].set_xlabel(None)
                ax_box[phase_index].set_ylabel('Slopes')
               
                annotator = Annotator(ax_box[phase_index], pairs, **plotting_parameters)
                annotator.configure(test=test_name, 
                                    loc='inside', 
                                    text_format='star', 
                                    show_test_name=True, 
                                    line_height=0.01, 
                                    fontsize=12, 
                                    color='gray', 
                                    verbose=False)           
                annotator.apply_and_annotate()
        plt.savefig(f'{SLEEP_ARCH_PLOTS_PATH}/slope_distribution_{test_name}.png')
        



### FUNCTION CALLING ###

### figure 3 ###

short_sleep = [103,'evening', '220717']
REM_rich = [103,'morning', '220718']
NREM_rich = [128,'baseline', '20221215']
wake_rich = [103, 'aversive', '220718']

args = [('REM_rich', REM_rich),
       ('NREM_rich', NREM_rich),
        ('Short_sleep', short_sleep),
        ('Wake_rich', wake_rich)]

get_any_plots_sleep_architecture_by_context(args)

### figure 4 ###

total_phases_boxplots('Kruskal', ME_norm_operation = None, sleep_length_norm = False)

### figure 5 ###

total_phases_boxplots('Kruskal', ME_norm_operation = 'div', sleep_length_norm = False)

### figure 6 ###

total_phases_boxplots('Kruskal', ME_norm_operation = 'div', sleep_length_norm = True)

### figure 7 ###

nums_boxplots('Kruskal', ME_norm_operation = None,  sleep_length_norm = False, data_folder= DATA_FOLDER)

### figure 8 ###

nums_boxplots('Kruskal', ME_norm_operation = 'div',  sleep_length_norm = False, data_folder= DATA_FOLDER)

### figure 9 ###

nums_boxplots('Kruskal', ME_norm_operation = 'div',  sleep_length_norm = True, data_folder= DATA_FOLDER)

### figure 10 ###

first_NREM_boxplots('Kruskal', ME_norm_operation = None , sleep_length_norm = False)

### figure 11 ###

first_NREM_boxplots('Kruskal', ME_norm_operation = 'div' , sleep_length_norm = False)

### figure 12 ###

first_NREM_boxplots('Kruskal', ME_norm_operation = 'div' , sleep_length_norm = True)

### figure 13 ###

get_scatter_plots('128', reg=True, only_fig=True)

### figure 14 ###

get_slope_boxplots(zero_test=True)

### figure 15 ###

get_slope_boxplots(scatter=True)

### figure 16 ###

get_slope_boxplots(slope_dist= True)




