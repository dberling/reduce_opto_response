from neurostim.analysis import quick_sim_setup
from neurostim.utils import convert_polar_to_cartesian_xz
from neurostim.analysis import get_AP_count
import numpy as np
import pandas as pd
from neuron import h

simcontrol = quick_sim_setup(
    cell_dict = dict(
        cellmodel=snakemake.wildcards.cell_id,
        ChR_soma_density=13e9,
        ChR_distribution='uniform'
    ),
    stimulator_dict = dict(
        diameter_um=200,
        NA=0.22
    )
)

thresh_int = np.load(str(snakemake.input[0]))

intensities = np.array(snakemake.params.rel_intensity) * thresh_int
radii_um = np.array([float(snakemake.wildcards.radius_um)])
angles_rad=np.array(snakemake.params.angles_rad)
temp_protocol=dict(
    duration_ms=200,
    delay_ms=1,
    total_rec_time_ms=400,
)
interpol_dt_ms=0.1

# load resistances:
node_x_resistance = np.load(snakemake.input[1])
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

results = []
APCs = []
for intensity in intensities:
    for radius in radii_um:
        for angle in angles_rad:
            stim_x_um, stim_y_um = convert_polar_to_cartesian_xz(radius, angle)
            stim_z_um = 0  # cortical surface
            tmp = simcontrol.run(
                temp_protocol=temp_protocol,
                stim_location=(stim_x_um, stim_y_um, stim_z_um),
                stim_intensity_mWPERmm2=intensity,
                rec_vars=rec_vars,
                interpol_dt_ms=interpol_dt_ms,
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
                    radius_um = radius,
                    angle_rad = angle,
                    intensity_mWPERmm2 = intensity,
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
            # annotate
            tmp['radius_um'] = radius
            tmp['angle_rad'] = angle
            tmp['intensity_mWPERmm2'] = intensity
            results.append(tmp)

pd.concat(results).to_csv(str(snakemake.output[0]))
pd.DataFrame(APCs).to_csv(str(snakemake.output[1]))
