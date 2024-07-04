from neurostim.stimulator import generate_occupied_positions
import pandas as pd

lp_configs = pd.read_csv('workflows/advanced_light_patterns/lp_configs.csv')
lp_config = lp_configs.loc[lp_configs.label == str(snakemake.wildcards.lp_config)].to_dict(orient='records')[0]
        
# Number of stimulators
N = int(lp_config['N']) 
# Distance between grid points in micrometers
grid_spacing = float(lp_config['p']) 
# Width of the grid abs and number of points along one dimension
abs_grid_width = 600 
grid_width = int(abs_grid_width / grid_spacing) + 1
# Diameter of the optical fiber
diameter_um = int(lp_config['d']) 
# Numerical aperture of the optical fiber
NA = float(lp_config['NA'])
# Intensity scale of each placed light source (constant across sources here)
intensity_scale = 1.0

occupied_positions = generate_occupied_positions(N, grid_width, grid_spacing)

stimulators_config = [
    {'diameter_um': diameter_um, 'NA': NA, 'position': pos, 'intensity_scale': intensity_scale}
    for pos in occupied_positions
]

pd.DataFrame(stimulators_config).to_csv(str(snakemake.output))
