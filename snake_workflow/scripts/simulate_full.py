from neurostim.analysis import quick_sim_setup_MultiStim
from neurostim.analysis import get_AP_count
import numpy as np
import pandas as pd
from neuron import h
import ast
import re
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

cell_dict = dict(
    cellmodel=str(snakemake.wildcards.cell_id),
    ChR_soma_density=13e9,
    ChR_distribution='uniform'
)
stimulator_config = pd.read_csv(
        str(snakemake.input), 
        index_col='Unnamed: 0'
        ).to_dict(orient='records')
# convert str(list) to real list:
for config in stimulator_config:
    config['position'] = ast.literal_eval(config['position'])

simcontrol = quick_sim_setup_MultiStim(cell_dict, stimulator_config)

temp_protocol=dict(
    duration_ms=200,
    delay_ms=1,
    total_rec_time_ms=500,
)
interpol_dt_ms=1

# Define recording variables
rec_vars = [[],[]]
# append time and soma voltage recoding
rec_vars[0].append('time [ms]')
rec_vars[1].append(h._ref_t)
rec_vars[0].append('v_soma_mV')
rec_vars[1].append(simcontrol.cell.model.soma_sec(0.5)._ref_v)

APCs = []
norm_power = float(snakemake.wildcards.norm_power)
try:
    tmp = simcontrol.run(
        temp_protocol=temp_protocol,
        stim_location=(0, 0, 0),
        stim_intensity_mWPERmm2=None,                    
        rec_vars=rec_vars,
        interpol_dt_ms=interpol_dt_ms,
        norm_power_mW_of_MultiStimulator=norm_power
    )
    # measure APC
    v_soma = tmp[['time [ms]', 'v_soma_mV']]
    v_soma_until_stim_period_stops = v_soma.loc[v_soma['time [ms]']<=temp_protocol['duration_ms']+temp_protocol['delay_ms']]
    x = tmp['time [ms]']
    y = tmp['v_soma_mV']
    peaks, properties = find_peaks(y, height=-20, prominence=10)
    APC = len(peaks)
    APCs.append(
        dict(
            simtype='full',
            patt_id = int(snakemake.wildcards.patt_id),
            norm_power_mW_of_MultiStimulator = norm_power,
            APC=APC
        )
    )

    if int(snakemake.wildcards.patt_id) % 10 == 0:
        plt.plot(x, y)
        plt.plot(x[peaks], y[peaks], 'ro', label='Detected Peaks')
        plt.legend()
        plt.savefig(str(snakemake.output)[:-4]+'_controlplot.png')

    del APC
except RuntimeError:
    print("RuntimeError at norm_power" +str(norm_power))

pd.DataFrame(APCs).to_csv(str(snakemake.output))
