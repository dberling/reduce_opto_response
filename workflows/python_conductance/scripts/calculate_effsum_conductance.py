import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from neurostim.chrimson_conductance_model import calc_rescaled_comp_conductances_nS

temp_protocol = dict(
    duration_ms = 200,
    delay_ms = 1,
    total_rec_time_ms=250
)

rescaled_comp_cond_nS = calc_rescaled_comp_conductances_nS(
    norm_power_mW_of_MultiStimulator=float(snakemake.wildcards.norm_power),
    stimulator_config = pd.read_csv(
        str(snakemake.input[0]),
        index_col='Unnamed: 0'
    ).to_dict(orient='records'),
    comp_data=np.load(str(snakemake.input[1])),
    temp_protocol = temp_protocol
) 

condsum = rescaled_comp_cond_nS.sum(axis=1)
# interpolate to 0.1 ms:
condsum = condsum[::int(len(condsum)/temp_protocol['total_rec_time_ms'])*10]

np.save(str(snakemake.output), condsum)
