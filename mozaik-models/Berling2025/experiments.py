#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.experiments.optogenetic import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
from copy import deepcopy


def central_stim_intensity_search(model,morphology,radius):
    trial = model.parameters["trial"]
    if radius == 50:
        if morphology == "None":
            intensities = [1.6]
        elif morphology == "m1":
            intensities = [0.2]
        elif morphology == "m2":
            intensities = [0.65]
    if radius == 150:
        if morphology == "None":
            intensities = [0.4]
        elif morphology == "m1":
            intensities = [0.054]
        elif morphology == "m2":
            intensities = [0.107]
    elif radius == 250:
        if morphology == "None":
            intensities = [0.299,0.298,0.2975,0.295,0.2925,0.29]
        elif morphology == "m1":
            intensities = [0.05]
        elif morphology == "m2":
            intensities = [0.09]
    return create_experiments_central_stimulation_morphology(model,morphology,radius,intensities[trial])

def central_stim_constant_fr(model,morphology):
    trial = model.parameters["trial"]
    radii = [50,150,250]
    if morphology == "None":
        intensities = [1.6,0.4,0.299]
    elif morphology == "m1":
        intensities = [0.2,0.054,0.05]
    elif morphology == "m2":
        intensities = [0.65,0.107,0.09]
    return create_experiments_central_stimulation_morphology(model,morphology,radii[trial],intensities[trial])

def central_stim_equal_input(model,morphology):
    trial = model.parameters["trial"]
    radii = [50,100,150,200,250]
    if morphology == "None":
        intensities = [0.0156,0.0039,0.001733,0.000975,0.000624]
    elif morphology == "m1":
        intensities = [0.2,0.05,0.0222,0.0125,0.008]
    elif morphology == "m2":
        intensities = [0.65,0.1625,0.0722,0.040625,0.026]
    return create_experiments_central_stimulation_morphology(model,morphology,radii[trial],intensities[trial])

def create_experiments_central_stimulation_morphology(model,morphology,radius,intensity):
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
        "num_trials": 30,
        "stimulator_array_parameters": MozaikExtendedParameterSet(
            {
                "size": 800,
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
                "coords": [[0,0]],
                "radius": radius,
                "intensity": intensity,
                "duration": 150,
                "onset_time": 0,
                "offset_time": 100,
            }
        ),
    }
    return [SingleOptogeneticArrayStimulus(model,MozaikExtendedParameterSet(p))]
