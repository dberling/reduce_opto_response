import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_fr_center(dfs, labels, title, **plot_kws):
    for df, label in zip(dfs, labels):
        if label != 'full':
            plt.plot(df.norm_power_mW_of_MultiStimulator, df.APC, label=label, **plot_kws)
        if label == 'full':
            plt.plot(df.norm_power_mW_of_MultiStimulator, df.APC, label=label, c='black', **plot_kws)
    plt.legend(title=title)
    return plt.gca()

def get_cond_scale_value(filename):
    # Check if 'full_active' is present in the filename
    if 'full_active' in filename:
        return 'full'
    
    # Regex to find the value after 'cond_scale_fct'
    match = re.search(r'cond_scale_fct([\d\.]+)', filename)
    
    if match:
        return match.group(1)
    
    # If no match is found
    return 'No cond_scale_fct or full_active found' 

APC_data = [
    pd.read_csv(f, index_col='Unnamed: 0') for f in snakemake.input
]
labels = [
    get_cond_scale_value(f) for f in snakemake.input
]
plot_fr_center(
    dfs=APC_data,
    labels=labels,
    title='cond_scale_factor',
    marker='.' 
)
plt.xscale('log')
plt.savefig(str(snakemake.output))
