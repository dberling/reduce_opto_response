from RONs.preparation import PrepareReducedOptogeneticNeurons

neuron_configs = [dict(neuron_type=str(snakemake.wildcards.cell_id),
                       ratio=1,
                       cond_scale_factor=float(snakemake.wildcards.cond_scale_fct),
                       ChR_distribution='uniform',
                       ChR_density=130,
                       grouping_filepath=str(snakemake.input))]

save_to = snakemake.output[0][:-15]
PrepareReducedOptogeneticNeurons(neuron_configs=neuron_configs, save_to_path=save_to)
