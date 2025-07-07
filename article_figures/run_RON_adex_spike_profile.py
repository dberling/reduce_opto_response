import os
os.chdir('/home/berling/reduce_opto_response/')
import numpy as np

# global variabl setup
neuron_model_name = "L23_PC_cADpyr229_1"
adex_fit = 'snake_workflow/adex_models/fit_to_L23_PC_cADpyr229_1.nestml'
ChR_soma_density=13e9
ChR_distribution='uniform'
stimulator_dict = dict(
        diameter_um=50,
        NA=0.1)
temp_protocol=dict(
    duration_ms=200,
    delay_ms=1,
    total_rec_time_ms=300,
    interpol_dt_ms=1
)
interpol_dt_ms = temp_protocol['interpol_dt_ms']

stim_intensity_mWPERmm2 = 0.2

driving_conductance_nS = np.load('RON_cond_reduced_calculation.npy')
driving_times_ms = np.arange(0, len(driving_conductance_nS))

# Run somatic-equivalent conductance at AdEx - use nest env
import numpy as np
import pandas as pd
import nest
from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils

module_name, neuron_model_name = NESTCodeGeneratorUtils.generate_code_for(adex_fit)

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

time_ms = driving_times_ms
# define/load driving stimulus
conductance_nS = driving_conductance_nS

# take time 0 away (nest does not like time 0):
time_ms = time_ms[1:]
conductance_nS = conductance_nS[1:]
# convert times to float as nest expects float not int
time_ms = time_ms.astype(float)

times, Vm, spike_times = nest_sim(time_ms, conductance_nS)

np.save('spike_times_RON_adex.npy', spike_times)
