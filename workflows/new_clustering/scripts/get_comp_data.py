from neurostim.analysis import quick_sim_setup
from neurostim.model_reduction import get_cell_data
import numpy as np

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
cell = simcontrol.cell

comp_data = get_cell_data(cell)

np.save(str(snakemake.output), comp_data)
