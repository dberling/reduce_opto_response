from neuron import h
from neurostim.cell import Cell
from neurostim import models
from neurostim.analysis import get_AP_count
import numpy as np
import pandas as pd
import ast

# load conductance sum from full model simulation
cond = pd.read_csv(str(snakemake.input[1]))
cond = cond.set_index(['norm_power_mW_of_MultiStimulator'])

# load normalization to account for subsampling of conductance locations
cond_norm_factor = np.load(str(snakemake.input[2]))

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

rel_intensities = np.array(snakemake.params.rel_intensity)
intensities = np.round(rel_intensities * thresh_int, 10)

# Define recording variables
rec_vars = [[],[]]
# append time and soma voltage recoding

APCs = []
for rel_intensity, intensity in zip(rel_intensities, intensities):
    for scale_fct in snakemake.params.cond_scale_factors:
        # define driving stimulus
        time_ms = cond.loc[intensity]['time [ms]'].values
        conductance_nS = cond.loc[intensity]['rescaled_cond_nS'].values.copy()
        # scale conductance according to compensate for subsampling of compartments
        conductance_nS *= float(cond_norm_factor)
        # scale conductance according to general scale factor
        conductance_nS *= float(scale_fct)
        # driving stimulus
        t = h.Vector(time_ms)
        y = h.Vector(conductance_nS)

        # rec variables
        rec_time = h.Vector().record(h._ref_t)
        rec_v = h.Vector().record(soma(0.5)._ref_v)

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
                lp_config = str(snakemake.wildcards.lp_config),
                patt_id = int(snakemake.wildcards.patt_id),
                cond_scale_factor = float(scale_fct),
                norm_power_mW_of_MultiStimulator = intensity,
                rel_intensity = rel_intensity,
                APC=APC,
                condsum = np.sum(conductance_nS)
            )
        )
pd.DataFrame(APCs).to_csv(str(snakemake.output[0]))
