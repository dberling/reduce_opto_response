import matplotlib.pyplot as plt
from mozaik.storage.datastore import PickledDataStore
from parameters import ParameterSet
from mozaik.storage.queries import *
import sys
import logging
import scipy
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from mozaik.analysis.analysis import Analysis
from mozaik.analysis.data_structures import SingleValue, AnalogSignal
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
import numpy as np
import mozaik.tools.units as munits
from neo.core.analogsignal import AnalogSignal as NeoAnalogSignal
import quantities as qt
from mozaik.storage import queries

class RecordingArrayAnalysis(Analysis):

    required_parameters = ParameterSet(
        {
            "s_res": int,  # Space resolution (bin size in um) of orientation map
            "array_width": float, # Electrode array width (um)
        }
    )
    
    def electrode_positions(self, array_width, s_res):
        assert array_width % s_res == 0
        row_electrodes = int(array_width / s_res)

        electrode_pos = np.linspace(
            s_res / 2, array_width - s_res / 2, row_electrodes
        )
        electrode_x, electrode_y = np.meshgrid(electrode_pos, electrode_pos)
        electrode_x, electrode_y = electrode_x.flatten(), electrode_y.flatten()
        return electrode_x - array_width / 2, electrode_y - array_width / 2

    def get_st_ids(self,dsv):
        assert len(dsv.sheets()) == 1, "Analysis needs to be run on a single sheet!"
        return [s for s in dsv.get_segments() if len(s.spiketrains) > 0][0].get_stored_spike_train_ids()

    def get_s(self, dsv, s_res=None, neuron_ids=None):
        if s_res == None:
            s_res = 1
        if neuron_ids is None:
            neuron_ids = self.get_st_ids(dsv)
        sheet = dsv.sheets()[0]
        pos = dsv.get_neuron_positions()[sheet]
        posx = np.round((pos[0, dsv.get_sheet_indexes(sheet, neuron_ids)] / s_res * 1000)).astype(int)
        posy = np.round((pos[1, dsv.get_sheet_indexes(sheet, neuron_ids)] / s_res * 1000)).astype(int)
        return posx, posy

    def neuron_electrode_dists(self, x, y, electrode_x, electrode_y):
        # Returns distance matrix (neurons x electrodes)
        neuron_x, neuron_y = (
            np.tile(x, (len(electrode_x), 1)).T,
            np.tile(y, (len(electrode_y), 1)).T,
        )
        electrode_x, electrode_y = np.tile(electrode_x, (len(x), 1)), np.tile(
            electrode_y, (len(y), 1)
        )
        return np.sqrt((electrode_x - neuron_x) ** 2 + (electrode_y - neuron_y) ** 2)
    
    def perform_analysis(self):
        self.tags.extend(["s_res: %d" % self.parameters.s_res,"array_width: %d" % self.parameters.array_width])

class RecordingArrayOrientationMap(RecordingArrayAnalysis):

    def gen_or_map(self, dsv, s_res, array_width):
        x, y = self.get_s(dsv)
        electrode_x, electrode_y = self.electrode_positions(array_width, s_res)
        d = self.neuron_electrode_dists(x, y, electrode_x, electrode_y)
        analysis_result = dsv.full_datastore.get_analysis_result(
            identifier="PerNeuronValue",
            value_name="LGNAfferentOrientation",
        )
        if len(analysis_result) == 0:
            NeuronAnnotationsToPerNeuronValues(dsv, ParameterSet({})).analyse()
        result = dsv.full_datastore.get_analysis_result(
            identifier="PerNeuronValue",
            value_name="LGNAfferentOrientation",
        )[0]
        st_ids = [s for s in dsv.get_segments() if len(s.spiketrains) > 0][
            0
        ].get_stored_spike_train_ids()
        orientations = np.array(result.get_value_by_id(st_ids))

        closest_neuron_idx = np.argmin(d,axis=0)#.astype(int)
        or_map_orientations = orientations[closest_neuron_idx]
        square_side = int(np.sqrt(len(or_map_orientations)))
        return or_map_orientations.reshape(square_side,square_side)

    def perform_analysis(self):
        super().perform_analysis()
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet)
            or_map = self.gen_or_map(dsv, self.parameters.s_res, self.parameters.array_width)  
            self.datastore.full_datastore.add_analysis_result(
                SingleValue(
                    value=or_map,
                    value_units=qt.radians,
                    value_name="orientation map",
                    tags=self.tags,
                    sheet_name=self.datastore.sheets()[0],
                    analysis_algorithm=self.__class__.__name__,
                )
            )
    
class RecordingArrayTimecourse(RecordingArrayAnalysis):
    
    required_parameters = ParameterSet(
        {
            "t_res": int,  # Time resolution (bin size in ms) of activity maps
            "s_res": int,  # Space resolution (bin size in um) of activity maps
            "array_width": float, # Electrode array width (um)
            "electrode_radius": float, # Electrode radius (um)
        }
    )
    
    def get_t(self, seg, t_res=None):
        if t_res == None:
            t_res = 1
        return [list((st.magnitude / t_res).astype(int)) for st in seg.get_spiketrains()]
    
    def neuron_spike_array(self, t, stim_len):
        s = np.zeros((len(t), int(stim_len)))
        for i in range(len(t)):
            for j in range(len(t[i])):
                if t[i][j] < stim_len:
                    s[i, t[i][j]] += 1
        return s

    def get_electrode_recordings(self, s, d, radius):
        # The recordings are a mean of all activity in the electrode radius
        rec = np.zeros((d.shape[1], s.shape[1]))
        for i in range(d.shape[1]):
            rec[i, :] += s[d[:, i] < radius, :].mean(axis=0)
        rec = rec.reshape(int(np.sqrt(d.shape[1])), int(np.sqrt(d.shape[1])), -1)
        return rec
    
    def perform_analysis(self):
        super().perform_analysis()
        self.tags.extend(["t_res: %d" % self.parameters.t_res, "electrode_radius: %d" % self.parameters.electrode_radius])
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet)
            x, y = self.get_s(dsv)
            segs, stims = dsv.get_segments(), dsv.get_stimuli()
            for i in range(len(segs)):
                if len(segs[i].spiketrains) < 1:
                    continue
                t = self.get_t(segs[i], t_res=self.parameters.t_res)
                stim_len = ParameterSet(str(stims[i]).replace("MozaikExtended",""))["duration"] // self.parameters.t_res
                electrode_x, electrode_y = self.electrode_positions(self.parameters.array_width, self.parameters.s_res)
                d = self.neuron_electrode_dists(x, y, electrode_x, electrode_y)
                s = self.neuron_spike_array(t, stim_len)
                electrode_recordings = np.nan_to_num(self.get_electrode_recordings(s / self.parameters.t_res * 1000, d, self.parameters.electrode_radius))
                electrode_recordings = electrode_recordings.transpose((2,0,1)) # Time should be first dimension
                self.datastore.full_datastore.add_analysis_result(
                    AnalogSignal(
                        NeoAnalogSignal(electrode_recordings, t_start=0, sampling_period=self.parameters.t_res*qt.ms,units=munits.spike / qt.s),
                        y_axis_units=munits.spike / qt.s,
                        tags=self.tags,
                        sheet_name=sheet,
                        stimulus_id=str(stims[i]),
                        analysis_algorithm=self.__class__.__name__,
                    )
                )

def retrieve_ds_param_values(dsv, param_name):
    # Hacky function because of DataStore limitations
    # Retrieves all direct stimulation parameter values from dsv
    l=[]
    for s in dsv.get_stimuli():
        if MozaikParametrized.idd(s).direct_stimulation_parameters != None:
            l.append(MozaikParametrized.idd(s).direct_stimulation_parameters.stimulating_signal_parameters[param_name])
    return sorted(list(set(l)))

def interpolate_2d(arr, target_shape):
    # Create a grid of coordinates for the input array
    N, M = arr.shape
    x_sparse, y_sparse = np.meshgrid(np.arange(M), np.arange(N))

    # Flatten the input array and coordinates
    x_sparse_flat = x_sparse.ravel()
    y_sparse_flat = y_sparse.ravel()
    arr_flat = arr.ravel()

    # Create a grid of coordinates for the target shape
    target_N, target_M = target_shape
    x_dense, y_dense = np.meshgrid(np.linspace(0, M-1, target_M), np.linspace(0, N-1, target_N))

    # Perform 2D interpolation using griddata
    z_dense_interpolated = scipy.interpolate.griddata((x_sparse_flat, y_sparse_flat), arr_flat, (x_dense, y_dense), method='linear')

    return z_dense_interpolated

def radial_mean(image, num_annuli):
    min_im_size = min(image.shape)
    image = image[image.shape[0]//2-min_im_size//2:image.shape[0]//2+min_im_size//2, image.shape[1]//2-min_im_size//2:image.shape[1]//2+min_im_size//2]
    if min_im_size // 2 != num_annuli:
        image = interpolate_2d(image, (num_annuli * 2,num_annuli * 2))
    center_x, center_y = num_annuli, num_annuli
    radius, angle = np.meshgrid(np.arange(num_annuli*2) - center_x, np.arange(num_annuli*2) - center_y, indexing='ij')
    radius = np.sqrt(radius**2 + angle**2)
    
    annulus_radii = np.linspace(0, num_annuli, num_annuli + 1)
    
    # Compute the average magnitude within each annulus
    radial_mean = np.zeros(num_annuli)
    for i in range(num_annuli):
        mask = (radius >= annulus_radii[i]) & (radius < annulus_radii[i + 1])
        radial_mean[i] = np.mean(np.abs(image[mask]))
    return radial_mean

paths_morph_1 = [
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250117-173530[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250120-144244[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250120-144250[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250120-144255[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250120-144301[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
]
paths_morph_2 = [
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250119-162430[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250120-145004[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250120-145010[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250120-145015[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250120-145020[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
    ]

paths_no_morph = [
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250120-213934[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250121-154310[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250121-154314[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250121-154319[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
    "/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250121-154324[param_spont.defaults]CombinationParamSearch{trial:[0]}/SelfSustainedPushPull_ParameterSearch_____trial:0",
]

paths_noconn_morph_1 = [
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214359[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214404[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214409[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214414[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214419[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
]

paths_noconn_morph_2 = [
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214426[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214431[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214436[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214440[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214445[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
]

paths_noconn_no_morph = [
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214312[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214317[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214322[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214327[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
    '/home/rozsa/dev/david_reduction_project/mozaik-models/LSV1M/20250127-214331[param_spont.defaults]CombinationParamSearch{11}/SelfSustainedPushPull_ParameterSearch_____base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_base_weight:0_store_stimuli:True',
]

ds_morph_1 = [
    PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": path, "store_stimuli": False}),
        replace=False,
    ) for path in paths_morph_1
]

ds_morph_2 = [
    PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": path, "store_stimuli": False}),
        replace=False,
    ) for path in paths_morph_2
]

ds_no_morph = [
    PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": path, "store_stimuli": False}),
        replace=False,
    ) for path in paths_no_morph
]

ds_noconn_morph_1 = [
    PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": path, "store_stimuli": False}),
        replace=False,
    ) for path in paths_noconn_morph_1
]

ds_noconn_morph_2 = [
    PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": path, "store_stimuli": False}),
        replace=False,
    ) for path in paths_noconn_morph_2
]

ds_noconn_no_morph = [
    PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": path, "store_stimuli": False}),
        replace=False,
    ) for path in paths_noconn_no_morph
]

t_res, s_res, array_width = 5, 50, 4000
if 1:
    for ds in ds_no_morph + ds_morph_1 + ds_morph_2 + ds_noconn_morph_1 + ds_noconn_morph_2 + ds_noconn_no_morph:
        ds.remove_ads_from_datastore()
        RecordingArrayTimecourse(param_filter_query(ds,sheet_name="V1_Exc_L2/3"),
            ParameterSet(
                {
                    "s_res": s_res,
                    "t_res": t_res,
                    "array_width": array_width,
                    "electrode_radius": 50,
                }
            ),
        ).analyse()
        ds.save()
