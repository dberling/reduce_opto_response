from neuron import h
from neurostim.cell import Cell
from neurostim import models
from neurostim.analysis import get_AP_count
import numpy as np
import pandas as pd
import ast

temp_protocol = dict(
    delay_ms = 1,
    total_rec_time_ms = 250
)
interpolt_dt_ms = 0.1

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
# define/load driving stimulus
conductance_nS = np.load(str(snakemake.input))
times_ms = np.arange(0,temp_protocol['total_rec_time_ms'],interpol_dt_ms)

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

h.v_init, h.tstop= -70, temp_protocol['total_rec_time_ms']
h.run()

# measure APC
APC = get_AP_count(
    df=pd.DataFrame(
        columns=['time [ms]','V_soma(0.5)'],
        data = np.array([rec_time, rec_v]).T
    ),
    interpol_dt_ms=interpol_dt_ms,
    t_on_ms=temp_protocol['delay_ms'],
    AP_threshold_mV=0
)
APCs.append(
    dict(
        lp_config = str(snakemake.wildcards.lp_config),
        patt_id = int(snakemake.wildcards.patt_id),
        cond_scale_factor = float(snakemake.wildcards.cond_scale_fct),
        norm_power_mW_of_MultiStimulator = float(snakemake.wildcards.norm_power),
        APC=APC,
        condsum = np.sum(conductance_nS)
    )
)
pd.DataFrame(APCs).to_csv(str(snakemake.output))
