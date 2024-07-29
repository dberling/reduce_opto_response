import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_full = pd.read_csv(str(snakemake.input[0]), index_col='Unnamed: 0')
df_effsum = pd.read_csv(str(snakemake.input[1], index_col='Unnamed: 0')
df = df_effsum.merge(df_full, on=['lp_config', 'patt_id', 'norm_power_mW_of_MultiStimulator'], suffixes=['_effsum', '_full'])

df['APC_error'] = (df.APC_effsum - df.APC_full).abs()
df['APC_MSE'] = (df.APC_effsum - df.APC_full)**2

df['APC>=1'] = df.APC_full
df.loc[df.APC_full == 0,'APC>=1'] = 1

df['rel_APC_error'] = df.APC_error / df['APC>=1']
df['APC_MSW/(APC>=1)^2'] = df.APC_MSE / df['APC>=1']**2

stats = df.groupby('cond_scale_factor')[['APC_error', 'APC_MSE', 'rel_APC_error', 'APC_MSW/(APC>=1)^2']].mean()
normed_MSE = df.groupby('cond_scale_factor')['APC_MSW/(APC>=1)^2'].mean()
np.save(str(snakemake.output[0]),normed_MSE.loc[normed_MSE==normed_MSE.min()].index.values)

fig, axs = plt.subplots(ncols=2)
stats[['APC_error', 'APC_MSE']].plot(kind='bar', ax=axs[0])
stats[['rel_APC_error', 'APC_MSW/(APC>=1)^2']].plot(kind='bar', ax=axs[1])
fig.savefig(str(snakemake.output[1]))
