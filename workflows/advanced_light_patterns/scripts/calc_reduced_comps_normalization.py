import pandas as pd
import numpy as np

data = pd.concat([pd.read_csv(fname, index_col=0) for fname in list(snakemake.input)])

summed_cond = data.groupby(['rec_cond_locs','norm_power_mW_of_MultiStimulator']).rescaled_cond_nS.sum().reset_index()
summed_cond['rec_cond_locs'] = summed_cond['rec_cond_locs'].astype(str)
pivot = summed_cond.pivot(index='norm_power_mW_of_MultiStimulator', columns='rec_cond_locs', values='rescaled_cond_nS')
for col in pivot.columns:
    # assume rec_cond_locs=='9999' means all compartments
    pivot[col] = pivot[col]/pivot['9999']

normalization_quotients = pivot.mean()

np.save(
    str(snakemake.output), 
    np.array(1/normalization_quotients[str(snakemake.wildcards.target_n_locs)])
)
