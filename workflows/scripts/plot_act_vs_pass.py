from neurostim.reduced_model_activation_plots import plot_active_vs_passive_cell_activation

fig, ax = plot_active_vs_passive_cell_activation(
        cell=snakemake.wildcards.cell_id, 
        path='spatial_data'
)
fig.savefig(str(snakemake.output), dpi=snakemake.params.dpi)
