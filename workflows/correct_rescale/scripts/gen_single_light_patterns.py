import pandas as pd


x = snakemake.params.x[int(snakemake.wildcards.patt_id)]
y = snakemake.params.y[int(snakemake.wildcards.patt_id)]
z = 0

stimulators_config = [{
    'diameter_um': float(snakemake.params.diameter_um), 
    'NA': float(snakemake.params.NA), 
    'position': [x, y, z], 
    'intensity_scale': 1.0
}]

pd.DataFrame(stimulators_config).to_csv(str(snakemake.output))
