from neuron import h
from neurostim.cell import Cell
from neurostim import models
from neurostim.analysis import get_AP_count
import numpy as np
import pandas as pd
import ast

# load conductance sum from full model simulation
cond = pd.read_csv(str(snakemake.input))
cond = cond.set_index(['norm_power_mW_of_MultiStimulator'])

intensities = cond.index.get_level_values('norm_power_mW_of_MultiStimulator').unique()

# set up neuron model
passive_cell_name = str(snakemake.wildcards.cell_id)
active_cell_name = passive_cell_name[:-10]
cellmodel = getattr(models, active_cell_name)
cell = Cell(
    model=cellmodel(),
    ChR_soma_density=13e9,
    ChR_distribution='uniform'
)
# insert new conductance to 'inject' conductance
soma = cell.sections[0]
soma.insert('g_chanrhod')

# Define recording variables
rec_vars = [[],[]]
# append time and soma voltage recoding

APCs = []
for intensity in intensities:
    # define driving stimulus
    time_ms = cond.loc[intensity]['time [ms]'].values
    conductance_nS = cond.loc[intensity]['rescaled_cond_nS'].values.copy()
    # scale conductance according to general scale factor
    conductance_nS *= float(snakemake.wildcards.cond_scale_fct)
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
            cond_scale_factor = float(snakemake.wildcards.cond_scale_fct),
            norm_power_mW_of_MultiStimulator = intensity,
            APC=APC,
            condsum = np.sum(conductance_nS)
        )
    )
pd.DataFrame(APCs).to_csv(str(snakemake.output[0]))
