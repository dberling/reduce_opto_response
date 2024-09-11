from neuron import h
from neurostim.cell import Cell
from neurostim import models
from neurostim.analysis import get_AP_count
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


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
with open(str(snakemake.input[1]), 'rb') as handle:
    temp_protocol = pickle.load(handle)
time_ms = np.arange(0,temp_protocol['total_rec_time_ms'],1) # conductance is saves at 1ms resolution
# conductance across compartments:
conds = np.load(str(snakemake.input[0]))
cond_soma = conds[:,0] # only soma
# scale with cond-scale-factor
conductance_nS = cond_soma * float(snakemake.wildcards.cond_scale_factor)

if (conductance_nS.shape == ()) and np.isnan(conductance_nS) == True:
    # calculation of conductance was rejected. Save dummy file.
    APCs.append(
        dict(
            patt_id = int(snakemake.wildcards.patt_id),
            norm_power_mW_of_MultiStimulator = float(snakemake.wildcards.norm_power),
            APC=np.nan
        )
    )
    pd.DataFrame(APCs).to_csv(str(snakemake.output))
else:
    # proceed with simulation

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

    # h.v_init, h.tstop= -70, temp_protocol['total_rec_time_ms']
    h.v_init, h.tstop= -70, 201
    h.run()

    # measure APC
    x = np.array(rec_time)
    y = np.array(rec_v)
    peaks, properties = find_peaks(y, height=-20, prominence=10)
    APC = len(peaks)
    APCs.append(
        dict(
            patt_id = int(snakemake.wildcards.patt_id),
            norm_power_mW_of_MultiStimulator = float(snakemake.wildcards.norm_power),
            APC=APC
        )
    )
    pd.DataFrame(APCs).to_csv(str(snakemake.output))

    if int(snakemake.wildcards.patt_id) % 10 == 0:
        plt.plot(x, y)
        plt.plot(x[peaks], y[peaks], 'ro', label='Detected Peaks')
        plt.legend()
        plt.savefig(str(snakemake.output)[:-4]+'_controlplot.png')
