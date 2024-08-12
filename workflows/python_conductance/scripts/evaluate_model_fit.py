import pandas as pd
import numpy as np

def find_idx_ymin_after_ymax(y):
    # Identify the index of the first peak
    first_peak_index = np.argmax(y)
    
    # Identify the index where the second increase starts
    second_increase_index = None
    for i in range(first_peak_index + 1, len(y) - 1):
        if y[i] < y[i + 1]:
            second_increase_index = i
            break
    return second_increase_index
def eval_model_performance(df_full, df_effsum, mode='until_full_max'):
    """
    Take AP counts in both models and compare:
    1. absolute spike deviation per paramset
    2. relative spike deviation, normalized by APC of full (or 1 if APC=0)

    Iterate over lp_config and patt_id and 
    exclude datapoints after the peak of the full model.
    """
    if mode == 'until_full_max':
        APC_deviation = []
        APC_dev_rel_to_full = []
        for lp_config in df_full.lp_config.unique():
            for patt_id in df_full.patt_id.unique():
                x= df_effsum.loc[(df_effsum.lp_config==lp_config) & (df_effsum.patt_id==patt_id)].norm_power_mW_of_MultiStimulator.values
                y= df_effsum.loc[(df_effsum.lp_config==lp_config) & (df_effsum.patt_id==patt_id)].APC.values
                idx_max = y.argmax()
                for APC_full, APC_effsum in zip(
                    df_full.loc[(df_full.lp_config==lp_config) & (df_full.patt_id==patt_id)].APC.values[:idx_max],
                    df_effsum.loc[(df_effsum.lp_config==lp_config) & (df_effsum.patt_id==patt_id)].APC.values[:idx_max]
                ):
                    APC_deviation.append(abs(APC_full-APC_effsum))
                    if APC_full == 0:
                        norm_APC = 1
                    else:
                        norm_APC = APC_full
                    APC_dev_rel_to_full.append(abs(APC_full-APC_effsum) / norm_APC)
    elif mode == 'exclude_effsum_dpb':
        APC_deviation = []
        APC_dev_rel_to_full = []
        for lp_config in df_full.lp_config.unique():
            for patt_id in df_full.patt_id.unique():
                x= df_effsum.loc[(df_effsum.lp_config==lp_config) & (df_effsum.patt_id==patt_id)].norm_power_mW_of_MultiStimulator.values
                y= df_effsum.loc[(df_effsum.lp_config==lp_config) & (df_effsum.patt_id==patt_id)].APC.values
                idx_min = find_idx_ymin_after_ymax(y)
                for APC_full, APC_effsum in zip(
                    df_full.loc[(df_full.lp_config==lp_config) & (df_full.patt_id==patt_id)].APC.values[:idx_min],
                    df_effsum.loc[(df_effsum.lp_config==lp_config) & (df_effsum.patt_id==patt_id)].APC.values[:idx_min]
                ):
                    APC_deviation.append(abs(APC_full-APC_effsum))
                    if APC_full == 0:
                        norm_APC = 1
                    else:
                        norm_APC = APC_full
                    APC_dev_rel_to_full.append(abs(APC_full-APC_effsum) / norm_APC)
    return [np.mean(APC_deviation), np.mean(APC_dev_rel_to_full), np.sqrt(np.mean(np.array(APC_deviation)**2)), np.sqrt(np.mean(np.array(APC_dev_rel_to_full)**2))]

files = list(snakemake.input)
full_model_files = [fname for fname in files if 'full_active_data' in fname]
effsum_model_files = [fname for fname in files if 'effsum_active_data' in fname]
df_full = pd.concat([pd.read_csv(fname, index_col=0) for fname in full_model_files])
df_full = df_full.sort_values(
    by=['lp_config', 'patt_id', 'norm_power_mW_of_MultiStimulator']) 
df_effsum = pd.concat([pd.read_csv(fname, index_col=0) for fname in effsum_model_files])
df_effsum = df_effsum.sort_values(
    by=['lp_config', 'patt_id', 'norm_power_mW_of_MultiStimulator']) 

mode = 'until_full_max'
adict = dict()
for label, result in zip(
    ['mean[APC_dev]', 'mean[rel_APC_dev]', 'sqrt(mean[APC_dev**2])', 'sqrt(mean[rel_APC_dev**2])'],
    eval_model_performance(df_full, df_effsum, mode=mode)
):
    adict[label] = result
pd.DataFrame([adict]).to_csv(str(snakemake.output[0]))

mode = 'exclude_effsum_dpb'
adict = dict()
for label, result in zip(
    ['mean[APC_dev]', 'mean[rel_APC_dev]', 'sqrt(mean[APC_dev**2])', 'sqrt(mean[rel_APC_dev**2])'],
    eval_model_performance(df_full, df_effsum, mode=mode)
):
    adict[label] = result
pd.DataFrame([adict]).to_csv(str(snakemake.output[1]))

for df, output in zip([df_full, df_effsum], [str(snakemake.output[2]), str(snakemake.output[3])]):
    np.save(
        output, 
        np.array(
            df.groupby(['lp_config', 'patt_id']).apply(lambda x: x.loc[x.APC == x.APC.max()].norm_power_mW_of_MultiStimulator).mean()
        )
    )
