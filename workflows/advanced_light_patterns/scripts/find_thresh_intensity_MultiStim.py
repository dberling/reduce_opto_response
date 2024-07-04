import numpy as np
import pandas as pd
import ast
from neurostim.analysis import quick_sim_setup_MultiStim
from neurostim.analysis import find_intensity_range

# cell parameters
cell_dict=dict(
    cellmodel=str(snakemake.wildcards.cell_id),
    ChR_soma_density=13e9, # equals 130 ch/um2
    ChR_distribution="uniform"
)
stimulator_config = pd.read_csv(
        str(snakemake.input), 
        index_col='Unnamed: 0'
        ).to_dict(orient='records')
# convert str(list) to real list:
for config in stimulator_config:
    config['position'] = ast.literal_eval(config['position'])

sim_control = quick_sim_setup_MultiStim(cell_dict, stimulator_config)

thresh_intensity = find_intensity_range(
    i_start_mWPERmm2 = 3e-6,
    sim_control=sim_control,
    temp_protocol=dict(
        duration_ms = 200,
        delay_ms = 50,
        total_rec_time_ms = 300
    ),
    analysis_params=dict(
        AP_threshold_mV = 0,
        interpol_dt_ms=0.1
    ),
    MultiStim=True
)
np.save(str(snakemake.output), np.array(thresh_intensity))
