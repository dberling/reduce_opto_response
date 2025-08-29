from neurostim.analysis import quick_sim_setup
from neurostim.model_reduction import assign_to_cartesian_grid, group_sections_by_proximity, transform_dict
import pickle
import matplotlib.pyplot as plt

# construct cell
simcontrol = quick_sim_setup(
    cell_dict=dict(
        cellmodel=str(snakemake.wildcards.cell_id),
        ChR_soma_density=13e9,
        ChR_distribution='uniform'
    ),
    stimulator_dict=dict(
        diameter_um=200,
        NA=0.22
    )
)
cell = simcontrol.cell

# cluster sections
cluster_secs = assign_to_cartesian_grid(
    cell.sections,
    grid_width=int(snakemake.wildcards.cluster_grid_width),
)

# group remaining sections to clustered ones
not_clustered_secs = [sec for sec in cell.sections if sec not in cluster_secs]
grouping, grouped_distances = group_sections_by_proximity(
    fixed_sections=cluster_secs,
    other_sections=not_clustered_secs
)
# tranform to dict with str instead nrn.section elements
grouping_str = transform_dict(grouping)
# save    
with open(str(snakemake.output[0]), 'wb') as handle:
    pickle.dump(grouping_str, handle)

plt.hist(grouped_distances)
plt.xlabel('distance [um]')
plt.ylabel('cnts')
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(str(snakemake.output[1]))

