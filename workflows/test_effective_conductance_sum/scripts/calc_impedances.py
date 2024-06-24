from neurostim.cell import Cell
from neurostim.analysis import quick_sim_setup
from neurostim.utils import convert_polar_to_cartesian_xz
from neurostim.analysis import get_AP_count
import numpy as np
import pandas as pd
from neuron import h
from neat import GreensTree
from neat import FourrierTools

simcontrol = quick_sim_setup(
    cell_dict = dict(
        cellmodel=str(snakemake.wildcards.cell_id),
        ChR_soma_density=13e9,
        ChR_distribution='uniform'
    ),
    stimulator_dict = dict(
        diameter_um=200,
        NA=0.22
    )
)
secs = simcontrol.cell.sections
allsegment_locs = []
for sec in secs:
    for seg in sec:
        allsegment_locs.append(
            dict(
                node=int(sec.name()),
                x=seg.x
            )
        )
# set up greenstree for impedance calculations        
greens_tree = simcontrol.cell.ph_tree.__copy__(new_tree=GreensTree())

# calc impedance kernels from 0 to 50 ms
# create a Fourriertools instance with the temporal array on which to evaluate the impedance kernel
t_arr = np.linspace(0.,200,4000)
ft = FourrierTools(t_arr)
# appropriate frequencies are stored in `ft.s`
# set the boundary condition for cylindrical segments in `greens_tree`
greens_tree.setImpedance(ft.s)# calc impedance kernels from 0 to 50 ms
# create a Fourriertools instance with the temporal array on which to evaluate the impedance kernel
t_arr = np.linspace(0.,200,4000)
ft = FourrierTools(t_arr)
# appropriate frequencies are stored in `ft.s`
# set the boundary condition for cylindrical segments in `greens_tree`
greens_tree.setImpedance(ft.s)

# record input resistances
data = []
for loc in allsegment_locs:
    # input resistance:
    ir = greens_tree.calcZF(loc, loc)[ft.ind_0s].real
    data.append([loc['node'], loc['x'], ir])

np.save(str(snakemake.output), data)