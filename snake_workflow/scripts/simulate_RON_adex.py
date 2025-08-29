import numpy as np
import pandas as pd
import pickle
import nest
from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils

module_name, neuron_model_name = NESTCodeGeneratorUtils.generate_code_for(str(snakemake.input[2]))

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

with open(str(snakemake.input[1]), 'rb') as handle:
    temp_protocol = pickle.load(handle)

time_ms = np.arange(0,temp_protocol['total_rec_time_ms'],temp_protocol['interpol_dt_ms'])

# define/load driving stimulus
conductance_nS = np.load(str(snakemake.input[0]))[0]

APCs = []
if (conductance_nS.shape == ()) and np.isnan(conductance_nS) == True:
    # calculation of conductance was rejected. Save dummy file.
    APCs.append(
        dict(
            simtype='RON_adex',
            patt_id = int(snakemake.wildcards.patt_id),
            cond_scale_factor = float(snakemake.wildcards.cond_scale_fct),
            norm_power_mW_of_MultiStimulator = float(snakemake.wildcards.norm_power),
            APC=np.nan
        )
    )
    pd.DataFrame(APCs).to_csv(str(snakemake.output))
else:
    # proceed with simulation

    # take time 0 away (nest does not like time 0):
    time_ms = time_ms[1:]
    conductance_nS = conductance_nS[1:]
    # convert times to float as nest expects float not int
    time_ms = time_ms.astype(float)

    times, Vm, spike_times = nest_sim(time_ms, conductance_nS)

    APCs.append(
        dict(
            simtype='RON_adex',
            patt_id = int(snakemake.wildcards.patt_id),
            cond_scale_factor = float(snakemake.wildcards.cond_scale_fct),
            norm_power_mW_of_MultiStimulator = float(snakemake.wildcards.norm_power),
            APC=len(spike_times)
        )
    )
    pd.DataFrame(APCs).to_csv(str(snakemake.output))
