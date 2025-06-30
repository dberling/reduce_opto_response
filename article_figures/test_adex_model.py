import os
os.chdir('/home/berling/reduce_opto_response/')
import pandas as pd
import numpy as np
import re
import pickle
import nest
from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils

nestml_neuron_path = 'snake_workflow/adex_models/fit_to_L23_PC_cADpyr229_1.nestml'

# load pre-simulated spatial activation maps
df = pd.read_csv('snake_workflow/result--L23_PC_cADpyr229_1_d50_NA0.1.csv', index_col='Unnamed: 0')
df_full = df.loc[(df.simtype=='full')]
df_RON_full = df.loc[(df.simtype=='RON_full')]
df_RON_adex = df.loc[(df.simtype=='RON_adex')]
merge = df_RON_full.merge(df_RON_adex, on=['patt_id','norm_power_mW_of_MultiStimulator', 'cond_scale_factor'])
merge = merge.loc[merge.patt_id==230]

def process_directory(directory):
    # Pattern with capture groups for patt_id and norm_power
    pattern = re.compile(
        r'slp-id230-xymax_110-dxy_10-norm_power_(?P<norm_power>[0-9eE\.-]+)-cluster_50-cond_scale_fct0\.85\.npy$'
    )

    results = []
    conds = dict()
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            file_path = os.path.join(directory, filename)
            try:
                array = np.load(file_path)
                if array.shape == (1, 250):
                    array_sum = array.sum()
                    norm_power = float(match.group('norm_power'))
                    conds[norm_power] = array[0]
                    results.append({
                        'cond': array_sum,
                        'norm_power': norm_power
                    })
                else:
                    print(f"Skipping {filename}: incorrect shape {array.shape}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    df = pd.DataFrame(results)
    return df, conds
df, conds = process_directory('snake_workflow/simulated_data/RON_cond/L23_PC_cADpyr229_1/')
df = df.rename(columns=dict(norm_power='norm_power_mW_of_MultiStimulator'))
cond_merge = merge.merge(df, on=['norm_power_mW_of_MultiStimulator'])

time_ms = np.arange(0,250)
conductance_nS = conds[list(conds.keys())[0]]

module_name, neuron_model_name = NESTCodeGeneratorUtils.generate_code_for(nestml_neuron_path)

def nest_sim(stim_times, stim_cond):
    #nest.set_verbosity("M_WARNING")
    #nest.set_verbosity("M_ERROR")
    nest.ResetKernel()

    nest.Install(module_name)
    
    neuron = nest.Create(neuron_model_name)
    
    voltmeter = nest.Create("voltmeter")
    voltmeter.set({"record_from": ["V_m"]})
    nest.Connect(voltmeter, neuron)
    
    sr = nest.Create("spike_recorder")
    nest.Connect(neuron, sr)
    
    scg = nest.Create('step_current_generator')
    
    nest.Connect(scg, neuron)
    
    scg.set({'amplitude_times':stim_times, 'amplitude_values':stim_cond,'stop':stim_times[-1]})
    
    nest.Simulate(stim_times[-1])

    return voltmeter.get("events")["times"], voltmeter.get("events")["V_m"], nest.GetStatus(sr, keys='events')[0]['times']


# take time 0 away (nest does not like time 0):
time_ms = time_ms[1:]
conductance_nS = conductance_nS[1:]
# convert times to float as nest expects float not int
time_ms = time_ms.astype(float)

APCs = []
cond_scales = np.arange(0.4,1.4,0.2)
for cond_scale in cond_scales:
    times, Vm, spike_times = nest_sim(time_ms, conductance_nS*cond_scale)
    APCs.append(len(spike_times))

np.save('test_adex_fit.npy',np.array([cond_scales, APCs]))
