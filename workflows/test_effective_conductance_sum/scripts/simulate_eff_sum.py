from neuron import h
from neurostim.cell import Cell
from neurostim import models
from neurostim.utils import convert_polar_to_cartesian_xz
from neurostim.analysis import get_AP_count
import numpy as np
import pandas as pd

# load conductance sum from full model simulation
cond = pd.read_csv(str(snakemake.input[1]))
cond = cond.set_index(['radius_um', 'angle_rad', 'intensity_mWPERmm2'])

# set up neuron model
cellmodel = getattr(models, str(snakemake.wildcards.cell_id))
cell = Cell(
    model=cellmodel(),
    ChR_soma_density=13e9,
    ChR_distribution='uniform'
)
# insert new conductance to 'inject' conductance
soma = cell.sections[0]
soma.insert('g_chanrhod')




thresh_int = np.load(str(snakemake.input[0]))

intensities = np.array(snakemake.params.rel_intensity) * thresh_int
radii_um = np.array([float(snakemake.wildcards.radius_um)])
angles_rad=np.array(snakemake.params.angles_rad)

# Define recording variables
rec_vars = [[],[]]
# append time and soma voltage recoding
rec_vars[0].append('time [ms]')
rec_vars[1].append(h._ref_t)
rec_vars[0].append('v_soma_mV')
rec_vars[1].append(soma(0.5)._ref_v)

APCs = []
for intensity in intensities:
    for radius in radii_um:
        for angle in angles_rad:
            stim_x_um, stim_y_um = convert_polar_to_cartesian_xz(radius, angle)
            stim_z_um = 0  # cortical surface

            # define driving stimulus
            time_ms = cond.loc[radius,angle,intensity]['time [ms]'].values
            conductance_nS = cond.loc[radius,angle,intensity]['rescaled_cond_nS'].values
            # driving stimulus
            t = h.Vector(time_ms)
            y = h.Vector(conductance_nS)

            # run simulation with injected conductance
            h.load_file('stdrun.hoc')

            # play the stimulus into soma(0.5)'s ina
            # the last True means to interpolate; it's not the default, but unless
            # you know what you're doing, you probably want to pass True there
            y.play(soma(0.5)._ref_gcat2_g_chanrhod, t, True)

            h.v_init, h.tstop= -70, 500
            h.run()

            # measure APC
            APC = get_AP_count(
                df=pd.DataFrame(
                    columns=['time [ms]','V_soma(0.5)'],
                    data = np.array([rec_time, rec_v]).T
                ),
                interpol_dt_ms=0.1,
                t_on_ms=1,
                AP_threshold_mV=0
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
