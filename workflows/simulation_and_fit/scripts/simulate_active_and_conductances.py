from neurostim.analysis import quick_sim_setup_MultiStim
from neurostim.analysis import get_AP_count
import numpy as np
import pandas as pd
from neuron import h
import ast
import re

passive_cell_name = str(snakemake.wildcards.cell_id)
active_cell_name = passive_cell_name[:-10]

cell_dict = dict(
    cellmodel=active_cell_name,
    ChR_soma_density=13e9,
    ChR_distribution='uniform'
)
stimulator_config = pd.read_csv(
        str(snakemake.input[1]), 
        index_col='Unnamed: 0'
        ).to_dict(orient='records')
# convert str(list) to real list:
for config in stimulator_config:
    config['position'] = ast.literal_eval(config['position'])

simcontrol = quick_sim_setup_MultiStim(cell_dict, stimulator_config)

intensities = np.array(snakemake.params.intensities)

temp_protocol=dict(
    duration_ms=200,
    delay_ms=1,
    total_rec_time_ms=500,
)
interpol_dt_ms=1

# load resistances:
node_x_resistance = np.load(snakemake.input[0])
# helper class to index sections as defined in node_x_resistance
class getSection():
    def __init__(self, sections):
        self.secs = np.array(sections)
        self.sec_idx = [int(str(sec)) for sec in sections]
    def __getitem__(self, key: int):
        return self.secs[self.sec_idx.index(key)]
getSecs = getSection(simcontrol.cell.sections)
# define recordings variables [names, pointers] according to uniformly distributed locations defined to calculate rescale factors:
comp_ids, conductance_pointers, resistances, seg_area_um2 = list(zip(*[('_'.join([str(int(node)),str(x)]), getSecs[int(node)](x)._ref_gcat_chanrhod, r, getSecs[int(node)](x).area() ) for (node, x, r) in node_x_resistance]))

# Define recording variables
rec_vars = list()
rec_vars.append(list(comp_ids))
rec_vars.append(list(conductance_pointers))
# append time and soma voltage recoding
rec_vars[0].append('time [ms]')
rec_vars[1].append(h._ref_t)
rec_vars[0].append('v_soma_mV')
rec_vars[1].append(simcontrol.cell.model.soma_sec(0.5)._ref_v)

# summarize compartment data in df:
comp_df= pd.DataFrame(
    columns=['comp', 'input_r', 'seg_area_um2'],
    data=zip(comp_ids, resistances, seg_area_um2)
)
comp_df['ir-ir_soma'] = comp_df["input_r"] - comp_df.loc[comp_df['comp']=='1_0.5']['input_r'].values[0]

condsum = []
cond_per_comp = []
APCs = []
for intensity in intensities:
        try:
            tmp = simcontrol.run(
                temp_protocol=temp_protocol,
                stim_location=(0, 0, 0),
                stim_intensity_mWPERmm2=None,                    
                rec_vars=rec_vars,
                interpol_dt_ms=interpol_dt_ms,
                norm_power_mW_of_MultiStimulator=intensity
            )
            # measure APC
            v_soma = tmp[['time [ms]', 'v_soma_mV']]
            v_soma_until_stim_period_stops = v_soma.loc[v_soma['time [ms]']<=temp_protocol['duration_ms']+temp_protocol['delay_ms']]
            APC = get_AP_count(
                df=v_soma_until_stim_period_stops,
                interpol_dt_ms=0.1,
                t_on_ms=1, AP_threshold_mV=0, apply_to="v_soma_mV"
            )
            APCs.append(
                dict(
                    lp_config = str(snakemake.wildcards.lp_config),
                    patt_id = int(snakemake.wildcards.patt_id),
                    norm_power_mW_of_MultiStimulator = intensity,
                    APC=APC
                )
            )
            # extract conductances:
            tmp = tmp.drop('v_soma_mV', axis=1)
            tmp = tmp.melt(id_vars=['time [ms]'], var_name='comp')
            tmp = tmp.rename(columns=dict(value='dens_cond_SPERcm2'))
            # merge seg data (diff of comp and soma input resistance and area)
            tmp = tmp.reset_index().merge(comp_df[['comp','ir-ir_soma', 'seg_area_um2']])
            # calculate compartment conductance from density conductance and comp area
            tmp['cond_nS'] = tmp['dens_cond_SPERcm2'] * 1e9 * tmp['seg_area_um2'] * (1e-4)**2
            # calculate recale_factor(conductance) times conductance
            tmp['rescaled_cond_nS'] = tmp['cond_nS'] / (1 + tmp['ir-ir_soma'] * tmp['cond_nS'])
             # delete irrelevant data
            tmp = tmp[['time [ms]', 'comp', 'rescaled_cond_nS']]
            # annotate
            df = tmp.copy()
            df['condition'] = '_'.join([
                str(snakemake.wildcards.lp_config),
                str(snakemake.wildcards.patt_id),
                str(intensity)
            ])
            df = df[['time [ms]', 'condition', 'comp', 'rescaled_cond_nS']]
            df = df.pivot(
                    index=['time [ms]', 'condition'], 
                    columns='comp', 
                    values='rescaled_cond_nS'
            ).reset_index()
            df = df.loc[df['time [ms]'] < 251]
            df.set_index(['time [ms]'], inplace=True)
            # integration over 10 ms rolling window
            window_size = int(10 / interpol_dt_ms)
            df = df.groupby(['condition']).rolling(window=window_size).sum()
            df.reset_index().set_index(['condition', 'time [ms]'], inplace=True)
            df= df * interpol_dt_ms
            # Downsample to 10ms steps
            df = df.reset_index().groupby('condition').apply(
                lambda frame: frame.iloc[window_size::window_size])
            df = df.drop(
                columns=['condition']
            ).reset_index(
            ).set_index(
                ['condition', 'time [ms]']
            ).drop(columns=['level_1'])
            cond_per_comp.append(df)

            # sum rescaled conductance over comps per time step
            tmpsum = pd.DataFrame(tmp.groupby('time [ms]')['rescaled_cond_nS'].sum()).rename(
                columns=dict(rescaled_cond_nS='rescaled_cond_nS')
            )
            # annotate
            tmpsum['lp_config'] = str(snakemake.wildcards.lp_config)
            tmpsum['patt_id'] = int(snakemake.wildcards.patt_id)
            tmpsum['norm_power_mW_of_MultiStimulator'] = intensity
            condsum.append(tmpsum)
        except RuntimeError:
            print("RuntimeError at intensity "+str(intensity))

pd.concat(cond_per_comp).to_csv(str(snakemake.output[2]))
pd.concat(condsum).to_csv(str(snakemake.output[1]))
pd.DataFrame(APCs).to_csv(str(snakemake.output[0]))
