import numpy as np
from neurostim.analysis import quick_sim_setup
from neurostim.analysis import find_intensity_range

# cell parameters
cell_dict=dict(
    cellmodel=str(snakemake.wildcards.cell_id),
    ChR_soma_density=13e9, # equals 130 ch/um2
    ChR_distribution="uniform"
)
stimulator_dict=dict(
    diameter_um=200, # diameter of fiber
    NA=0.22, # numerical aperture of fiber
)

sim_control = quick_sim_setup(cell_dict, stimulator_dict)

thresh_intensity = find_intensity_range(
    i_start_mWPERmm2 = 1e-4,
    sim_control=sim_control,
    temp_protocol=dict(
        duration_ms = 200,
        delay_ms = 50,
        total_rec_time_ms = 300
    ),
    analysis_params=dict(
        AP_threshold_mV = 0,
        interpol_dt_ms=0.1
    )
)
np.save(str(snakemake.output), np.array(thresh_intensity))
