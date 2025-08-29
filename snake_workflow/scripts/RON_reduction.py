from RONs.reduction import CreateReducedOptogeneticNeuron
import pickle

# spatially reduce neuron with grouping onto cartesian grid
Reducer = CreateReducedOptogeneticNeuron(snakemake.wildcards.cell_id)
grouping = Reducer.create_grouping(cluster_grid_width=snakemake.wildcards.cluster_grid_width)
print("Grouped compartments onto ", len(list(grouping.keys())), " nodes.")

with open(snakemake.output, 'wb') as file:
    pickle.dump(grouping, file)
