from RONs.reduction import CreateReducedOptogeneticNeuron
import pickle
import numpy as np

# spatially reduce neuron with grouping onto cartesian grid
Reducer = CreateReducedOptogeneticNeuron(snakemake.wildcards.cell_id)
grouping = Reducer.create_grouping(cluster_grid_width=snakemake.wildcards.cluster_grid_width)
print("Grouped compartments onto ", len(list(grouping.keys())), " nodes.")

with open(snakemake.output[0], 'wb') as file:
    pickle.dump(grouping, file)

np.save(snakemake.output[1], Reducer.soma_sec_coords)
