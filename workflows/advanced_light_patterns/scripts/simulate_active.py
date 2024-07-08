from neurostim.analysis import quick_sim_setup_MultiStim
from neurostim.analysis import get_AP_count
import numpy as np
import pandas as pd
from neuron import h
import ast

passive_cell_name = str(snakemake.wildcards.cell_id)
active_cell_name = passive_cell_name[:-10]

cell_dict = dict(
    cellmodel=active_cell_name,
    ChR_soma_density=13e9,
    ChR_distribution='uniform'
)
stimulator_config = pd.read_csv(
        str(snakemake.input[1]), 
        index_col='Unnamed: 0'
        ).to_dict(orient='records')
# convert str(list) to real list:
for config in stimulator_config:
    config['position'] = ast.literal_eval(config['position'])

simcontrol = quick_sim_setup_MultiStim(cell_dict, stimulator_config)

thresh_int = np.load(str(snakemake.input[0]))

rel_intensities = np.round(np.array(snakemake.params.rel_intensity) * thresh_int, 10)
intensities = np.round(rel_intensities * thresh_int, 10)

temp_protocol=dict(
    duration_ms=200,
    delay_ms=1,
    total_rec_time_ms=500,
)
interpol_dt_ms=0.1

# Define recording variables
rec_vars = [[],[]]
# append time and soma voltage recoding
rec_vars[0].append('time [ms]')
rec_vars[1].append(h._ref_t)
rec_vars[0].append('v_soma_mV')
rec_vars[1].append(simcontrol.cell.model.soma_sec(0.5)._ref_v)

APCs = []
for rel_intensity, intensity in zip(rel_intensities, intensities):
    tmp = simcontrol.run(
        temp_protocol=temp_protocol,
        stim_location=(0, 0, 0),
        stim_intensity_mWPERmm2=None,
        rec_vars=rec_vars,
        interpol_dt_ms=interpol_dt_ms,
        norm_power_mW_of_MultiStimulator=intensity
    )
    # measure APC
    v_soma = tmp[['time [ms]', 'v_soma_mV']]
    v_soma_until_stim_period_stops = v_soma.loc[v_soma['time [ms]']<=temp_protocol['duration_ms']+temp_protocol['delay_ms']]
    APC = get_AP_count(
        df=v_soma_until_stim_period_stops,
        interpol_dt_ms=0.1,
        t_on_ms=1, AP_threshold_mV=0, apply_to="v_soma_mV"
    )
    APCs.append(
        dict(
            lp_config = str(snakemake.wildcards.lp_config),
            patt_id = int(snakemake.wildcards.patt_id),
            norm_power_mW_of_MultiStimulator = intensity,
            rel_intensity = rel_intensity,
            APC=APC
        )
    )

pd.DataFrame(APCs).to_csv(str(snakemake.output[0]))
