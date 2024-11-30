import numpy as np
import nest
import pickle
import pandas as pd
from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils
import re

module_name, neuron_model_name = NESTCodeGeneratorUtils.generate_code_for(
    str(snakemake.input.neuron_model)
)

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

APCs = []
for temp_protocol_file, conductance_file in zip(
        snakemake.input.temp_protocols, snakemake.input.conductances):
    # Regular expressions to extract the required values
    slp_id_pattern = r"slp-id(\d+)"
    norm_power_pattern = r"norm_power_([\d.]+)"
    # Extract values using regex
    slp_id_match = re.search(slp_id_pattern, conductance_file)
    norm_power_match = re.search(norm_power_pattern, conductance_file)
    # Check if matches were found and extract values
    slp_id = int(slp_id_match.group(1)) if slp_id_match else None
    norm_power = float(norm_power_match.group(1)) if norm_power_match else None

    with open(str(temp_protocol_file), 'rb') as handle:
        temp_protocol = pickle.load(handle)
    time_ms = np.arange(0,temp_protocol['total_rec_time_ms'],temp_protocol['interpol_dt_ms'])

    # define/load driving stimulus
    conductance_nS = np.load(str(conductance_file))

    if (conductance_nS.shape == ()) and np.isnan(conductance_nS) == True:
        # calculation of conductance was rejected. Save dummy file.
        APCs.append(
            dict(
                patt_id = int(slp_id),
                cond_scale_factor = float(snakemake.wildcards.cond_scale_fct),
                norm_power_mW_of_MultiStimulator = float(norm_power),
                APC=np.nan
            )
        )
    else:
        # proceed with simulation
        # scale conductance according to general scale factor
        conductance_nS *= float(snakemake.wildcards.cond_scale_fct)

        # take time 0 away (nest does not like time 0):
        time_ms = time_ms[1:]
        conductance_nS = conductance_nS[1:]

        times, Vm, spike_times = nest_sim(time_ms, conductance_nS)

        APCs.append(
            dict(
                patt_id = int(slp_id),
                cond_scale_factor = float(snakemake.wildcards.cond_scale_fct),
                norm_power_mW_of_MultiStimulator = float(norm_power),
                APC=len(spike_times)
            )
        )

pd.DataFrame(APCs).to_csv(str(snakemake.output))
