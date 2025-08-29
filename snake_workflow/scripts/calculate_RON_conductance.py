import ast
import pandas as pd
import numpy as np
import pickle
from RONs import ChR_dynamics
from RONs import LoadReducedOptogeneticNeurons
from neurostim.stimulator import MultiStimulator


temp_protocol = dict(
    duration_ms = 200,
    delay_ms = 1,
    total_rec_time_ms=250,
    interpol_dt_ms = 1
)
# define light flux over time and space
def flux(x, y, z):
    # set up stimulator
    stimulator_config = pd.read_csv(
        str(snakemake.input[3]),
        index_col='Unnamed: 0'
    ).to_dict(orient='records')[0],
    # convert str(list) to real list if needed:
    if type(stimulator_config[0]['position']) == str:
        for config in stimulator_config:
            try:
                config['position'] = ast.literal_eval(config['position'])
            except ValueError:
                print(config['position'])
                raise ValueError("config['position'] raised ValueError")
    from neurostim.model_reduction import calc_fluxes_photons_PER_cm2_fs
    fluxes = calc_fluxes_photons_PER_cm2_fs(
        comp_xyz=[x,y,z],
        stimulator=MultiStimulator(stimulator_config),
        norm_power_mW_of_MultiStimulator=float(snakemake.wildcards.norm_power)
    )
    times = np.arange(0,temp_protocol['total_rec_time_ms'],temp_protocol['interpol_dt_ms'])
    on = np.zeros(shape=times.shape)
    on[int(temp_protocol['delay_ms']/temp_protocol['interpol_dt_ms']):int(
        (temp_protocol['delay_ms']+temp_protocol['duration_ms'])/temp_protocol['interpol_dt_ms'])
       ] = 1
    fluxes = [flux * on for flux in fluxes]
    return np.array(fluxes)

x_soma, y_soma, z_soma = np.load(snakemake.input[4]) 

RONs = LoadReducedOptogeneticNeurons(
        point_neuron_positions=np.array([[x_soma],[y_soma],[z_soma]]), # point neurons soma coordinates
        rotation_angles=np.array([0]), # random rotations of the morphology in radian
        neuron_cond_scale_path=snakemake.input[0],
        neuron_node_data_path=snakemake.input[1], # where to load "node data" of neuron morphologies
        neuron_comp_data_path=snakemake.input[2]  # where to load "comp data" of neuron morphologies
    )
RONs.calc_light_fluxes(
        func_xyz_to_flux=flux, 
        func_args=dict()
)
# init ChR
ChR_model = ChR_dynamics()
# calculate effective optogenetic somatic conductance
RONs.calc_effective_somatic_ChR_conductance(
    func_flux_to_conductance=ChR_model.calculate, 
    func_args=dict(update_interval=temp_protocol['interpol_dt_ms'])
)
# save conductance
np.save(str(snakemake.output[0]), RONs.conductance_nS_over_time)
# save temp_protocol
with open(str(snakemake.output[1]), 'wb') as handle:
    pickle.dump(temp_protocol, handle)
