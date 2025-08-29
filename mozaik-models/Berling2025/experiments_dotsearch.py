#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.experiments.optogenetic import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
from copy import deepcopy

def create_experiments_central_stimulation_morphology(model,morphology):
    trial = model.parameters["trial"]
    if morphology == "None":
        intensity = 1.6
    elif morphology == "m1":
        intensity = 0.2
    elif morphology == "m2":
        intensity = 0.65
    morphology_p = {
        "None": None,
        "m1": ParameterSet({
            "neuron_node_data_path": "L23_PC_cADpyr229_1_passdends-cluster_50-cond_scl_0.15-_node_data.npy",
            "neuron_comp_data_path": "L23_PC_cADpyr229_1_passdends-cluster_50-cond_scl_0.15-_comp_data.npy",
            "neuron_cond_scale_path": "L23_PC_cADpyr229_1_passdends-cluster_50-cond_scl_0.15-_cond_scale.npy",
            "soma_depth": 424,
            "min_max_smplngstep_morphology_range": (0, 800, 5),
            "ChR_calc_RAM_reduction_split": 2,
        }),
        "m2": ParameterSet({
            "neuron_node_data_path": "L23_PC_cADpyr229_3_passdends-cluster_50-cond_scl_0.1-_node_data.npy",
            "neuron_comp_data_path": "L23_PC_cADpyr229_3_passdends-cluster_50-cond_scl_0.1-_comp_data.npy",
            "neuron_cond_scale_path": "L23_PC_cADpyr229_3_passdends-cluster_50-cond_scl_0.1-_cond_scale.npy",
            "soma_depth": 390,
            "min_max_smplngstep_morphology_range": (0, 800, 5),
            "ChR_calc_RAM_reduction_split": 2,
        }),
    }
    p = {
        "sheet_list": ["V1_Exc_L2/3"],
        'sheet_intensity_scaler': [1.0],
        'sheet_transfection_proportion': [1.0],
        "num_trials": 1,
        "stimulator_array_parameters": MozaikExtendedParameterSet(
            {
                "size": 4000,
                "spacing": 10,
                "depth_sampling_step": 10,
                "light_source_light_propagation_data": "light_scattering_radial_profiles_lsd10.pickle",
                "update_interval": 1,
                "morphology": morphology_p[morphology],
            }
        ),
        "stimulating_signal": "mozaik.sheets.direct_stimulator.stimulating_pattern_flash",
        "stimulating_signal_parameters": ParameterSet(
            {
                "shape": "circle",
                "coords": [[0,0],[int(trial),0]],
                "radius": 50,
                "intensity": intensity,
                "duration": 150,
                "onset_time": 0,
                "offset_time": 100,
            }
        ),
    }
    return [SingleOptogeneticArrayStimulus(model,MozaikExtendedParameterSet(p))]
