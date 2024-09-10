import pandas as pd

x = float(snakemake.wildcards.x_coord)
y = float(snakemake.wildcards.y_coord)
z = 0

stimulators_config = [
    {'diameter_um': 200, 'NA': 0.22, 'position': [x, y, z], 'intensity_scale': 1.0}
]

pd.DataFrame(stimulators_config).to_csv(str(snakemake.output))
