from neurostim.analysis import quick_sim_setup
from neurostim.utils import convert_polar_to_cartesian_xz
from neurostim.analysis import get_AP_count
import numpy as np
import pandas as pd
from neuron import h

passive_cell_name = str(snakemake.wildcards.cell_id)
active_cell_name = passive_cell_name[:-10]

simcontrol = quick_sim_setup(
    cell_dict = dict(
        cellmodel=active_cell_name,
        ChR_soma_density=13e9,
        ChR_distribution='uniform'
    ),
    stimulator_dict = dict(
        diameter_um=200,
        NA=0.22
    )
)

thresh_int = np.load(str(snakemake.input[0]))

intensities = np.round(np.array(snakemake.params.rel_intensity) * thresh_int, 8)
radii_um = np.array([float(snakemake.wildcards.radius_um)])
angles_rad=np.array(snakemake.params.angles_rad)
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
for intensity in intensities:
    for radius in radii_um:
        for angle in angles_rad:
            stim_x_um, stim_y_um = convert_polar_to_cartesian_xz(radius, angle)
            stim_z_um = 0  # cortical surface
            tmp = simcontrol.run(
                temp_protocol=temp_protocol,
                stim_location=(stim_x_um, stim_y_um, stim_z_um),
                stim_intensity_mWPERmm2=intensity,
                rec_vars=rec_vars,
                interpol_dt_ms=interpol_dt_ms,
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
                    radius_um = radius,
                    angle_rad = angle,
                    intensity_mWPERmm2 = intensity,
                    APC=APC
                )
            )

pd.DataFrame(APCs).to_csv(str(snakemake.output[0]))
