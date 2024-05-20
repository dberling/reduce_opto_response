import numpy as np
import pandas as pd
from neurostim.analysis import simulate_spatial_profile
from neurostim.analysis import get_AP_count

# analysis params:
interpol_dt_ms = 0.1 # interpolate simulation results
AP_threshold_mV = 0  # used to threshold to count spikes
radii_um=np.arange(0, 550, 50)
angles_rad=np.arange(0, 2*np.pi, 2*np.pi/16)
# temporal stimulation protocol
temp_protocol = dict(
    duration_ms = 200, # stim duration
    delay_ms = 50, # start of stimulation
    total_rec_time_ms = 300 # simulation time
)
# cell parameters
if snakemake.wildcards.dendstate == 'passive':
    cellmodel = str(snakemake.wildcards.cell_id) + '_passdends'
elif snakemake.wildcards.dendstate == 'active':
    cellmodel = str(snakemake.wildcards.cell_id)
else:
    raise ValueError("dendstate must be 'active' or 'passive'")

cell_dict=dict(
    cellmodel=cellmodel,
    ChR_soma_density=13e9, # equals 130 ch/um2
    ChR_distribution="uniform"
)
stimulator_dict=dict(
    diameter_um=200, # diameter of fiber
    NA=0.22, # numerical aperture of fiber
)
thresh_intensity_mWPERmm2 = np.load(str(snakemake.input))
stim_intensity_mWPERmm2 = thresh_intensity_mWPERmm2 * float(snakemake.wildcards.rel_stim_intensity)

def analyze_AP_count(sim_data, segs):
    return get_AP_count(
            sim_data, 
            interpol_dt_ms=interpol_dt_ms,
            t_on_ms=temp_protocol['delay_ms'],
            AP_threshold_mV=AP_threshold_mV
    )

# run simulation of spatial profile
results = simulate_spatial_profile(
    cell_dict=cell_dict,
    stimulator_dict=stimulator_dict,
    stim_intensity_mWPERmm2=stim_intensity_mWPERmm2,
    radii_um=radii_um,
    angles_rad=angles_rad,
    temp_protocol=temp_protocol,
    seg_rec_vars=None,
    allseg_rec_var=None,
    sim_data_transform=None,
    scalar_result_names = ['AP_count'],
    scalar_result_funcs = [analyze_AP_count],
    vector_result_func = None,
    interpol_dt_ms=interpol_dt_ms,
    AP_threshold_mV=AP_threshold_mV
)
results.to_csv(str(snakemake.output))
