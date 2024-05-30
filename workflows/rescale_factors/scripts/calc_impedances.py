from neurostim.cell import Cell
from neurostim import models
from neat import GreensTree
from neat import FourrierTools
import numpy as np

cellmodel = getattr(models,str(snakemake.wildcards.cell_id))
cell_dict = dict(
    model=cellmodel(),
    ChR_soma_density=13e9,
    ChR_distribution='uniform'
)
cell = Cell(**cell_dict)

ph_tree = cell.ph_tree
uni_locs = ph_tree.distributeLocsUniform(dx=int(snakemake.params.dx))


greens_tree = ph_tree.__copy__(new_tree=GreensTree())

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
for loc in uni_locs:
    # input resistance:
    ir = greens_tree.calcZF(loc, loc)[ft.ind_0s].real
    data.append([loc['node'], loc['x'], ir])

np.save(str(snakemake.output), data)
