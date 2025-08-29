from mozaik.analysis.analysis import Analysis
from mozaik.analysis.data_structures import SingleValue, AnalogSignal
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
import numpy as np
import numpy
import mozaik.tools.units as munits
from neo.core.analogsignal import AnalogSignal as NeoAnalogSignal
import quantities as qt
from mozaik.storage import queries
from mozaik.storage.datastore import DataStoreView  
import scipy
from skimage.draw import disk
import copy
import copy
from som import SOM
from mozaik.visualization.plotting import Plotting
import pylab
from parameters import ParameterSet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import logging
import scipy

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

class SimulatedCalciumSignal(Analysis):
    
    required_parameters = ParameterSet(
        {
            "spatial_profile_path": str,  # numpy array
            "reference_dsv": DataStoreView,
        }
    )
    
    def calcium_light_spread_kernel(self,spatial_profile_path,s_res, array_width):
        x,y,I = np.load(spatial_profile_path)
        x_t, y_t = np.meshgrid(np.arange(-array_width//2,array_width//2+1,s_res),np.arange(-array_width//2,array_width//2+1,s_res))
        return scipy.interpolate.griddata((x.flatten(), y.flatten()), I.flatten(), (x_t, y_t), method='cubic')
    
    def t_kernel(self, t_res,length_ms=5000):
        # Based on https://doi.org/10.3389/fncir.2013.00201
        tau_on = 10 # ms rise time of calcium response
        tau_off = 1000 # ms fall time of calcium response
        if length_ms < 10 * t_res:
            length_ms = 10*t_res

        # We ignore the rise time for the moment as it is 100x smaller than fall time
        return np.exp(-np.linspace(0,length_ms,length_ms//t_res)/tau_off)

    def get_calcium_signal(self, A_in, t_res, s_res, array_width):
        A = A_in.copy()
        t_ker = self.t_kernel(t_res)
        A_t = scipy.ndimage.convolve1d(A, t_ker, axis=0, mode='wrap', origin=-(len(t_ker)//2))
        s_ker = self.calcium_light_spread_kernel(self.parameters.spatial_profile_path,s_res,array_width)
        A = np.stack([scipy.signal.convolve(A_t[i,:,:],s_ker,mode='same') for i in range(A.shape[0])])
        return A

    def normalize_calcium_signal(self, A, A_ref):
        tiling = np.ones_like(A.shape)
        tiling[0] = A.shape[0]
        f0 = np.tile(A_ref.mean(axis=0)[np.newaxis,...],tiling)
        return (A - f0) / f0
    
    def tag_value(self, tag, tags):
        filtered_tags = [t.split(":")[-1] for t in tags if ":".join(t.split(":")[:-1]) == tag]
        assert len(filtered_tags) == 1, "Duplicate tags are not allowed!"
        return eval(filtered_tags[0])
    
    def perform_analysis(self):
        datastore_equal_to_ref_dsv = self.datastore == self.parameters.reference_dsv
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet,analysis_algorithm="RecordingArrayTimecourse")
            for anasig in dsv.get_analysis_result():                
                t_res, s_res, array_width = self.tag_value("t_res",anasig.tags), self.tag_value("s_res",anasig.tags), self.tag_value("array_width",anasig.tags)
                assert anasig.analog_signal.ndim == 3, "Analog signal should have 1 temporal and 2 spatial dimensions!"
                calcium_signal = self.get_calcium_signal(anasig.analog_signal, t_res, s_res, array_width)
                if not datastore_equal_to_ref_dsv:
                    ref_dsv = queries.param_filter_query(self.parameters.reference_dsv, sheet_name=sheet,y_axis_name="Calcium imaging signal")
                    assert len(ref_dsv.analysis_results) == 1, "Reference datastore must contain exactly 1 Calcium imaging signal per sheet, contains %d" % len(ref_dsv.analysis_results)
                    ref_anasig = ref_dsv.analysis_results[0]
                    assert ref_anasig.analog_signal.ndim == 3, "Reference analog signal should have 1 temporal and 2 spatial dimensions!"
                    assert t_res == self.tag_value("t_res",ref_anasig.tags) and s_res == self.tag_value("s_res",ref_anasig.tags), "Reference and analysis analog signal need to be of the same spatial and temporal resolution!"
                    ref_calcium_signal = np.array(ref_anasig.analog_signal)
                else:
                    ref_calcium_signal = calcium_signal
                calcium_signal_normalized = self.normalize_calcium_signal(calcium_signal, ref_calcium_signal)
                common_params = {
                    "y_axis_units": qt.dimensionless,
                    "tags": anasig.tags,
                    "sheet_name": sheet,
                    "stimulus_id": anasig.stimulus_id,
                    "analysis_algorithm": self.__class__.__name__,
                }
                self.datastore.full_datastore.add_analysis_result(
                    AnalogSignal(
                        NeoAnalogSignal(calcium_signal, t_start=0, sampling_period=t_res*qt.ms,units=qt.dimensionless),
                        y_axis_name="Calcium imaging signal",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    AnalogSignal(
                        NeoAnalogSignal(calcium_signal, t_start=0, sampling_period=t_res*qt.ms,units=qt.dimensionless),
                        y_axis_name="Calcium imaging signal (normalized)",
                        **common_params,
                    )
                )

class GaussianBandpassFilter(Analysis):
    
    required_parameters = ParameterSet(
        {
            "highpass_sigma_um": float,
            "lowpass_sigma_um": float,
        }
    )

    def tag_value(self, tag, tags):
        if len(tags) == 0:
            raise RuntimeError("No tags on recording!")
        filtered_tags = [t.split(":")[-1] for t in tags if ":".join(t.split(":")[:-1]) == tag]
        assert len(filtered_tags) == 1, "Duplicate tags are not allowed!"
        return eval(filtered_tags[0])
    
    def bandpass_filter(self, A_in, high_sigma_um, low_sigma_um, s_res):
        hp_sigma = high_sigma_um / s_res
        lp_sigma = low_sigma_um / s_res

        # Band-pass filtering
        filt = np.zeros(len(A_in.shape))
        filt[1:] = lp_sigma
        flp = scipy.ndimage.gaussian_filter(A_in, filt)
        filt[1:] = hp_sigma
        fhp = scipy.ndimage.gaussian_filter(flp, filt)
        fbp = flp - fhp 

        return fbp
    
    def perform_analysis(self):
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet,identifier="AnalogSignal")
            for anasig in dsv.get_analysis_result():
                s_res, t_res = self.tag_value("s_res",anasig.tags), self.tag_value("t_res",anasig.tags)
                bpf = self.bandpass_filter(anasig.analog_signal, self.parameters.highpass_sigma_um, self.parameters.lowpass_sigma_um, s_res)
                self.datastore.full_datastore.add_analysis_result(
                    AnalogSignal(
                        NeoAnalogSignal(bpf, t_start=0, sampling_period=t_res*qt.ms,units=qt.dimensionless),
                        y_axis_units=qt.dimensionless,
                        tags=anasig.tags,
                        sheet_name=sheet,
                        stimulus_id=anasig.stimulus_id,
                        analysis_algorithm=self.__class__.__name__,
                    )
                )

class CorrelationMaps(Analysis):
    required_parameters = ParameterSet({})
    def correlation_maps(self,A):
        Av = (A - A.mean(axis=0)).transpose(1,2,0)
        Avss = (Av * Av).sum(axis=2)
        return np.array([np.nan_to_num(np.matmul(Av,Av[x,y,:])/ np.sqrt(Avss[x,y] * Avss)) for x in range(Av.shape[0]) for y in range(Av.shape[1])])
    
    def perform_analysis(self):
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet,identifier="AnalogSignal")
            for anasig in dsv.get_analysis_result():
                assert anasig.analog_signal.ndim == 3, "Signal must have 1 temporal and 2 spatial dimensions!"
                cm = self.correlation_maps(np.array(anasig.analog_signal))
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=cm,
                        value_units=qt.dimensionless,
                        value_name="correlation map",
                        tags=[t for t in anasig.tags if "t_res" not in t],
                        sheet_name=sheet,
                        stimulus_id=anasig.stimulus_id,
                        analysis_algorithm=self.__class__.__name__,
                    )
                )

class OrientationMapSimilarity(Analysis):
    required_parameters = ParameterSet({
        "or_map_dsv": DataStoreView,
    })

    def or_map_similarity(self,A,or_map):
        s_map = np.zeros(A.shape[0])
        or_map_s = np.sin(or_map).flatten()
        or_map_c = np.cos(or_map).flatten()
        for i in range(A.shape[0]):
            r_s = np.nan_to_num(scipy.stats.pearsonr(A[i,:,:].flatten(),or_map_s)[0])
            r_c = np.nan_to_num(scipy.stats.pearsonr(A[i,:,:].flatten(),or_map_c)[0])
            s_map[i] = np.sqrt(r_s*r_s + r_c*r_c)
        return s_map
        
    def perform_analysis(self):
        or_map_dsv_res = queries.param_filter_query(self.parameters.or_map_dsv,analysis_algorithm="RecordingArrayOrientationMap").get_analysis_result()
        assert len(or_map_dsv_res) == 1, "or_map_dsv can only contain 1 RecordingArrayOrientationMap per sheet, contains %d" % len(or_map_dsv_res)
        or_map = or_map_dsv_res[0].value
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet)
            for res in dsv.get_analysis_result():
                tags = res.tags
                stimulus_id = res.stimulus_id
                if type(res) == AnalogSignal:
                    res = np.array(res.analog_signal)
                elif type(res) == SingleValue:
                    res = np.array(res.value)
                assert res.ndim == 3, "Signal must have 1 arbitrary and 2 spatial dimensions!"
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=self.or_map_similarity(res,or_map),
                        value_units=qt.dimensionless,
                        value_name="orientation map similarity",
                        tags=tags,
                        sheet_name=sheet,
                        stimulus_id=stimulus_id,
                        analysis_algorithm=self.__class__.__name__,
                    )
                )

class CorrelationMapSimilarity(Analysis):
    required_parameters = ParameterSet({
        "corr_map_dsv": DataStoreView,
        "exclusion_radius": float,
    })

    def tag_value(self, tag, tags):
        if len(tags) == 0:
            raise RuntimeError("No tags on recording!")
        filtered_tags = [t.split(":")[-1] for t in tags if ":".join(t.split(":")[:-1]) == tag]
        assert len(filtered_tags) == 1, "Duplicate tags are not allowed!"
        return eval(filtered_tags[0])
    
    def correlation_map_similarity(self,C1,C2,s_res,exclusion_radius=400):
        assert C1[0].shape == C2[0].shape
        s_map = np.zeros(C1.shape[0])
        
        for i in range(C1.shape[0]):
            x,y = i // C1.shape[1], i % C1.shape[1]
            rr, cc = disk((x,y), exclusion_radius // s_res,shape=C1[i].shape)
            inv = np.zeros_like(C1[i])
            inv[rr,cc] = 1
            s_map[i], _ = scipy.stats.pearsonr(C1[i][inv < 1],C2[i][inv < 1])
        return s_map
        
    def perform_analysis(self):
        c1_dsv = queries.param_filter_query(self.datastore,analysis_algorithm="CorrelationMaps")
        c2_dsv = queries.param_filter_query(self.parameters.corr_map_dsv,analysis_algorithm="CorrelationMaps")

        # Find all unique pairs of correlation maps        
        unique_pairs = []
        for sheet_1 in c1_dsv.sheets():
            for sheet_2 in c2_dsv.sheets():
                c1_dsv_sh = queries.param_filter_query(c1_dsv,sheet_name=sheet_1)
                c2_dsv_sh = queries.param_filter_query(c2_dsv,sheet_name=sheet_2)
                for c1_res in c1_dsv_sh.get_analysis_result():
                    for c2_res in c2_dsv_sh.get_analysis_result():
                        if c1_res == c2_res:
                            continue
                        unique_pairs.append([c1_res,c2_res])

        unique_pairs = [sorted(p,key=lambda s: str(s)) for p in unique_pairs]
        unique_pairs = {str(p) : p for p in unique_pairs}
        unique_pairs = [unique_pairs[k] for k in set(unique_pairs.keys())]

        for c1_res, c2_res in unique_pairs:
            assert type(c1_res) == type(c2_res) == SingleValue 
            s_res = self.tag_value("s_res", c1_res.tags)
            assert c1_res.value.ndim == 3 and c2_res.value.ndim == 3, "Correlation maps must have 1 arbitrary and 2 spatial dimensions!"
            self.datastore.full_datastore.add_analysis_result(
                SingleValue(
                    value=self.correlation_map_similarity(c1_res.value, c2_res.value, s_res,self.parameters.exclusion_radius),
                    value_units=qt.dimensionless,
                    value_name="correlation map similarity",
                    tags=c1_res.tags,
                    stimulus_id=c1_res.stimulus_id,
                    analysis_algorithm=self.__class__.__name__,
                )
            )


class Smith_2018_Mulholland_2021_2024_spont_analyses(Analysis):
    required_parameters = ParameterSet({
    })

    def tag_value(self, tag, tags):
        if len(tags) == 0:
            raise RuntimeError("No tags on recording!")
        filtered_tags = [t.split(":")[-1] for t in tags if ":".join(t.split(":")[:-1]) == tag]
        if len(filtered_tags) == 0:
            return None
        assert len(filtered_tags) == 1, "Duplicate tags are not allowed!"
        return eval(filtered_tags[0])

    def fit_spatial_scale_correlation(self, distances, correlations):
        decay_func = lambda x,xi,c0 : np.exp(-x/xi) * (1-c0) + c0
        (xi, c0), _ = scipy.optimize.curve_fit(
            f=decay_func,
            xdata=distances,
            ydata=correlations,
            p0=[1,0],
        )
        return xi
        
    def find_local_maxima(self, arr, min_dist):
        xmax0, ymax0 = scipy.signal.argrelextrema(arr,np.greater_equal,order=min_dist,axis=0)
        xmax1, ymax1 = scipy.signal.argrelextrema(arr,np.greater_equal,order=min_dist,axis=1)
        s1 = {(xmax0[i],ymax0[i],arr[xmax0[i],ymax0[i]]) for i in range(len(xmax0))}
        s2 = {(xmax1[i],ymax1[i],arr[xmax1[i],ymax1[i]]) for i in range(len(xmax1))}
        s = sorted(list(s1 & s2),key=lambda el : el[2],reverse=True)
        i = 0
        while i < len(s):
            j=i+1
            while j < len(s):
                if (s[i][0] - s[j][0])**2 + (s[i][1] - s[j][1])**2 < min_dist**2:
                    s.pop(j)
                else:
                    j+=1
            i+=1
        return s
    
    def chance_similarity(or_map, s_res, t_res, coords):
        # Calculate similarity chance level
        # Generate 120 s of white noise activity, run through calcium imaging pipeline
        # Calculate self.tag_value("s_res",anasig.tags) maps and their similarity to orientation map
        random_act = np.dstack([np.random.rand(len(or_map.flatten())).reshape((or_map.shape[0],or_map.shape[1])) for i in range(2400)])
        random_act = bandpass_filter(get_calcium_signal(random_act,s_res,t_res),s_res)
        random_corr = correlation_maps(random_act, coords)
        return correlation_or_map_similarity(random_corr, coords, or_map).flatten()
    
    def local_maxima_distance_correlations(self, Cmaps,  s_res):
        cs, ds = [], []
        for i in range(Cmaps.shape[0]):
            coords = np.array([i // Cmaps.shape[1], i % Cmaps.shape[1]])
            min_distance_between_maxima = 800 #um
            maxima = np.array(self.find_local_maxima(Cmaps[i,:,:],min_distance_between_maxima//s_res))[:,:2].astype(int)
            d = np.sqrt(np.sum((maxima - coords)**2,axis=1)) * s_res / 1000
            c = Cmaps[i][maxima[:,0],maxima[:,1]]
            order = np.argsort(d)
            ds.append(d[order])
            cs.append(c[order])
        return np.array([np.hstack(ds),np.hstack(cs)])

    def interpolate_2d(self, arr, target_shape):
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
    
    def radial_mean(self, image, num_annuli):
        min_im_size = min(image.shape)
        image = image[image.shape[0]//2-min_im_size//2:image.shape[0]//2+min_im_size//2, image.shape[1]//2-min_im_size//2:image.shape[1]//2+min_im_size//2]
        if min_im_size // 2 != num_annuli:
            image = self.interpolate_2d(image, (num_annuli * 2,num_annuli * 2))
        center_x, center_y = num_annuli, num_annuli
        radius, angle = np.meshgrid(np.arange(num_annuli*2) - center_x, np.arange(num_annuli*2) - center_y, indexing='ij')
        radius = np.sqrt(radius**2 + angle**2)
        
        annulus_radii = np.linspace(0, num_annuli, num_annuli + 1)
        
        # Compute the average magnitude within each annulus
        radial_mean = np.zeros(num_annuli)
        for i in range(num_annuli):
            mask = (radius >= annulus_radii[i]) & (radius < annulus_radii[i + 1])
            radial_mean[i] = np.mean(image[mask])
        return radial_mean
        
    def corr_wavelength(self, Cmaps,s_res,array_width):
        select_size_um = 2500
        sel_min = select_size_um // 2 // s_res
        sel_max = array_width // s_res - sel_min
        sel_sz =  2 * sel_min
        coords = [(x,y) for x in range(Cmaps.shape[1]) for y in range(Cmaps.shape[2])]
        mean_Cmap = np.array([Cmaps[i,coords[i][0]-sel_sz//2:coords[i][0]+sel_sz//2,coords[i][1]-sel_sz//2:coords[i][1]+sel_sz//2] for i in range(len(coords)) if coords[i][0] > sel_min and coords[i][1] > sel_min and coords[i][0] < sel_max and coords[i][1] < sel_max]).mean(axis=0)
        num_annuli = 200
        rmean = self.radial_mean(mean_Cmap, num_annuli)
        wavelength = scipy.signal.argrelmax(rmean,order=10)[0][-1] * (sel_sz * s_res // 2) / num_annuli
        return wavelength / 1000

    def activity_wavelength(self,autocorr_rmeans,s_res):
        wls = np.linspace(0,autocorr_rmeans.shape[1]*s_res / 1000,autocorr_rmeans.shape[1])
        indices = []
        for i in range(autocorr_rmeans.shape[0]):
            arm = scipy.signal.argrelmin(autocorr_rmeans[i,:])
            if len(arm) > 0 and len(arm[0]) > 0:
                indices.append(arm[0][0])
        return wls[indices] * 2
    
    def autocorrelation_radial_mean(self,events):
        boundary = 5 # cut off the sides to avoid errors from autocorrelation wrap
        events = events[:,boundary:-boundary,boundary:-boundary]
        events = (events.transpose((1,2,0)) / np.linalg.norm(events,axis=(1,2)))
        autocorrs = np.stack([scipy.signal.correlate2d(events[:,:,i], events[:,:,i],mode='full',boundary='wrap') for i in range(events.shape[2])])
        autocorrs = autocorrs[:,events.shape[0]//4:-events.shape[0]//4,events.shape[1]//4:-events.shape[1]//4]
        return np.array([self.radial_mean(autocorrs[i,:,:], autocorrs.shape[1] // 2) for i in range(autocorrs.shape[0])])
    
    def modularity(self,autocorr):
        modularity = []
        for i in range(autocorr.shape[0]):
            armin = scipy.signal.argrelmin(autocorr[i,:],order=5)[0][0] # First minimum
            armax = scipy.signal.argrelmax(autocorr[i,armin:],order=5)[0][0] + armin # First maximum after the minimum
            modularity.append(np.abs(autocorr[i,armin] - autocorr[i,armax]))
        return np.array(modularity)
    
    def cart_to_pol(self, coeffs):
        """
    
        Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
        ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
        The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
        ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
        respectively; e is the eccentricity; and phi is the rotation of the semi-
        major axis from the x-axis.
    
        """
    
        # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
        # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
        # Therefore, rename and scale b, d and f appropriately.
        a = coeffs[0]
        b = coeffs[1] / 2
        c = coeffs[2]
        d = coeffs[3] / 2
        f = coeffs[4] / 2
        g = coeffs[5]
    
        den = b**2 - a*c
        if den > 0:
            raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                             ' be negative!')
    
        # The location of the ellipse centre.
        x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den
    
        num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        fac = np.sqrt((a - c)**2 + 4*b**2)
        # The semi-major and semi-minor axis lengths (these are not sorted).
        ap = np.sqrt(num / den / (fac - a - c))
        bp = np.sqrt(num / den / (-fac - a - c))
    
        # Sort the semi-major and semi-minor axis lengths but keep track of
        # the original relative magnitudes of width and height.
        width_gt_height = True
        if ap < bp:
            width_gt_height = False
            ap, bp = bp, ap
    
        # The eccentricity.
        r = (bp/ap)**2
        if r > 1:
            r = 1/r
        e = np.sqrt(1 - r)
    
        # The angle of anticlockwise rotation of the major-axis from x-axis.
        if b == 0:
            phi = 0 if a < c else np.pi/2
        else:
            phi = np.arctan((2.*b) / (a - c)) / 2
            if a > c:
                phi += np.pi/2
        if not width_gt_height:
            # Ensure that phi is the angle to rotate to the semi-major axis.
            phi += np.pi/2
        phi = np.real(phi) % np.pi
    
        return x0, y0, ap, bp, e, phi
    
    def fit_ellipse(self, x, y):
        """
    
        Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
        the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
        arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].
    
        Based on the algorithm of Halir and Flusser, "Numerically stable direct
        least squares fitting of ellipses'.
    
    
        """
    
        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones(len(x))]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        T = -np.linalg.inv(S3) @ S2.T
        M = S1 + S2 @ T
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.inv(C) @ M
        eigval, eigvec = np.linalg.eig(M)
        con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
        ak = eigvec[:, np.nonzero(con > 0)[0]]
        return np.concatenate((ak, T @ ak)).ravel()
    
    def local_correlation_eccentricity(self, Cmaps, margin=0):
        eccentricities = []
        for i in range(Cmaps.shape[0]):
            x,y = Cmaps.shape[0] // Cmaps.shape[1], Cmaps.shape[0] % Cmaps.shape[1]
            if x < margin or y < margin or x > Cmaps.shape[1] - margin or y > Cmaps.shape[2] - margin:
                continue
            C = Cmaps[i]
    
            # Crop the image to just the ellipse to make it faster!
            lw, num = scipy.ndimage.label(C>0.7)
            lw -= scipy.ndimage.binary_erosion(lw)
            X, Y = np.where(lw)
    
            try:
                coeffs = self.fit_ellipse(X, Y)
            except:
                break
            if len(coeffs) == 6:
                x0, y0, ap, bp, e, phi = self.cart_to_pol(coeffs)
                eccentricities.append(e)
    
        return np.array(eccentricities)

    def dim_random_sample_events(self, events, samples=30, repetitions=100):
        return np.array([self.dimensionality(events[np.random.choice(range(events.shape[2]),samples),:,:]) for i in range(repetitions)])

    def percentile_thresh(self, A, percentile):
        A_sorted = copy.deepcopy(A)
        A_sorted.sort(axis=0)
        thresh_idx = int(np.round((A.shape[0] - 1) * percentile))
        if len(A_sorted.shape) == 1:
            return A_sorted[thresh_idx]
        elif len(A_sorted.shape) == 3:
            return A_sorted[thresh_idx, :, :]
        else:
            return None

    def extract_event_indices(
        self, A, t_res, px_active_p=0.995, event_activity_p=0.8, min_segment_duration=100
    ):
        thresh = self.percentile_thresh(A, px_active_p)
        A_active = A.copy()
        A_active[A_active < thresh] = 0
        A_active[A_active >= thresh] = 1
        A_active_sum = A_active.sum(axis=(1, 2))
    
        thresh = self.percentile_thresh(A_active_sum, event_activity_p)
        A_active_zeroed = A_active_sum.copy()
        A_active_zeroed[A_active_zeroed < thresh] = 0
    
        segment_indices = []
        i = 0
        while i < A.shape[0]:
            if A_active_zeroed[i] > 0:
                segment_max = 0
                segment_max_idx = 0
                segment_start = i
                while A_active_zeroed[i] != 0:
                    if A_active_zeroed[i] > segment_max:
                        segment_max_idx = i
                        segment_max = A_active_zeroed[i]
                    i += 1
                    if i >= A.shape[0] - 1:
                        break
                if i - segment_start > min_segment_duration // t_res:
                    segment_indices.append(i)
            i += 1
        
        return segment_indices
    
    def kohonen_map(self,Cmaps,or_map):
        som = SOM(1,40,sigma_start=40)
        data = np.array([C.flatten() for C in Cmaps])
        som.fit(data,epochs=1000,verbose=False)
        return self.vector_readout(som,or_map)
    
    def find_ideal_rotation(self,ref,rot,rot_min,rot_max,n_steps=1000):
        ref = (ref-ref.min()) / (ref.max() - ref.min())
        rot = (rot-rot_min) / (rot_max - rot_min)
        steps = np.linspace(0,1,n_steps)
        best_step = 0
        best_err = np.inf
        best_rot = None
        for step in steps:
            err = ((ref-np.fmod(rot+step,1))**2).sum()
            if err < best_err:
                best_step = step
                best_err = err
                best_rot = np.fmod(rot+step,1)
        for step in steps:
            err = ((ref-np.fmod(1-rot+step,1))**2).sum()
            if err < best_err:
                best_step = step
                best_err = err
                best_rot = np.fmod(1-rot+step,1)
        return best_rot*(rot_max-rot_min) + rot_min
    
    def vector_readout(self,som,or_map):
        nodes = som.map.squeeze()
        angles = np.linspace(0,np.pi * 2,nodes.shape[0],endpoint=False)
        v = nodes.T * np.exp(1j * angles)
        v = (np.angle(v.sum(axis=1)).reshape((100,100)) + np.pi) / 2
        return self.find_ideal_rotation(or_map,v,0,np.pi)
    
    def circ_dist(a, b):
        return np.pi/2 - abs(np.pi/2 - abs(a-b))

    def resize_arr(self, A, new_width, new_height):
        A = np.asarray(A)
        shape = list(A.shape)
        shape[1] = new_width
        shape[2] = new_height
        ind = np.indices(shape, dtype=float)
        ind[1] *= (A.shape[1] - 1) / float(new_width - 1)
        ind[2] *= (A.shape[2] - 1) / float(new_height - 1)
        return scipy.ndimage.interpolation.map_coordinates(A, ind, order=1)
        
    def dimensionality(self, A):
        A = A.transpose((1,2,0))
        A = A.reshape((-1,A.shape[2]))
        try:
            cov_mat = numpy.cov(A)
            e = np.linalg.eigvalsh(cov_mat)
        except Exception as e:
            print(e)
            return -1
        return e.sum()**2 / (e*e).sum() 
    
    def perform_analysis(self):
        r = {
            "A_exc_calcium": queries.param_filter_query(self.datastore,y_axis_name="Calcium imaging signal (normalized)",sheet_name="V1_Exc_L2/3",st_name='InternalStimulus'),
            "A_inh_calcium": queries.param_filter_query(self.datastore,y_axis_name="Calcium imaging signal (normalized)",sheet_name="V1_Inh_L2/3",st_name='InternalStimulus'),
            "A_exc_bandpass": queries.param_filter_query(self.datastore,analysis_algorithm="GaussianBandpassFilter",sheet_name="V1_Exc_L2/3",st_name='InternalStimulus'),
            "A_inh_bandpass": queries.param_filter_query(self.datastore,analysis_algorithm="GaussianBandpassFilter",sheet_name="V1_Inh_L2/3",st_name='InternalStimulus'),
            # TODO: Maybe add case for raw correlation maps?
            "Cmaps_exc": queries.param_filter_query(self.datastore,analysis_algorithm="CorrelationMaps",sheet_name="V1_Exc_L2/3",st_name='InternalStimulus'),
            "Cmaps_inh": queries.param_filter_query(self.datastore,analysis_algorithm="CorrelationMaps",sheet_name="V1_Inh_L2/3",st_name='InternalStimulus'),
            "or_map": queries.param_filter_query(self.datastore,analysis_algorithm="RecordingArrayOrientationMap",sheet_name="V1_Exc_L2/3"),
        }

        tags = r["A_exc_calcium"].get_analysis_result()[0].tags
        stimulus_id = r["A_exc_calcium"].get_analysis_result()[0].stimulus_id
        s_res, t_res, array_width = self.tag_value("s_res", tags), self.tag_value("t_res", tags), self.tag_value("array_width", tags)
        
        for k in r.keys():
            ar = r[k].get_analysis_result()
            assert len(ar) == 1, "Can only contain single analysis result per sheet, contains %d" % len(ar)
            assert s_res == self.tag_value("s_res", ar[0].tags) and array_width == self.tag_value("array_width", ar[0].tags)
            if self.tag_value("t_res", ar[0].tags) is not None:
                assert t_res == self.tag_value("t_res", ar[0].tags)
            r[k] = ar[0].value if type(ar[0]) == SingleValue else ar[0].analog_signal
            r[k] = np.array(r[k])

        results = {}
    
        # Smith 2018
        event_idx_exc = self.extract_event_indices(r["A_exc_calcium"],t_res)
        event_idx_inh = self.extract_event_indices(r["A_inh_calcium"],t_res)
        small_spont_exc = self.resize_arr(r["A_exc_bandpass"][event_idx_exc,:,:], 50, 50)
        small_spont_inh = self.resize_arr(r["A_inh_bandpass"][event_idx_inh,:,:], 50, 50)
        results["Dimensionality"] = {"V1_Exc_L2/3" : self.dimensionality(small_spont_exc)}
        autocorr_radial_means = self.autocorrelation_radial_mean(r["A_exc_bandpass"][event_idx_exc,:,:])
        results["Event activity wavelength"] =  {"V1_Exc_L2/3" : self.activity_wavelength(autocorr_radial_means,s_res)}
        results["Event activity modularity"] =  {"V1_Exc_L2/3" : self.modularity(autocorr_radial_means)}                               
        
        results["Local maxima distance correlation"] = {"V1_Exc_L2/3" : self.local_maxima_distance_correlations(r["Cmaps_exc"], s_res),
                                                        "V1_Inh_L2/3" : self.local_maxima_distance_correlations(r["Cmaps_inh"], s_res)}
        results["Spatial scale of correlation"] = {"V1_Exc_L2/3" : self.fit_spatial_scale_correlation(results["Local maxima distance correlation"]["V1_Exc_L2/3"][0,:],
                                                                                                      results["Local maxima distance correlation"]["V1_Exc_L2/3"][1,:])}

        # Mulholland 2021
        results["Correlation map wavelength"] = {"V1_Exc_L2/3" : self.corr_wavelength(r["Cmaps_exc"],s_res,array_width),
                                                 "V1_Inh_L2/3" : self.corr_wavelength(r["Cmaps_inh"],s_res,array_width)}
        results["Local correlation eccentricity"] =  {"V1_Exc_L2/3" : self.local_correlation_eccentricity(r["Cmaps_exc"]),
                                                      "V1_Inh_L2/3" : self.local_correlation_eccentricity(r["Cmaps_inh"])}
        results["Dimensionality (random sampled events)"] =  {"V1_Exc_L2/3" : self.dim_random_sample_events(small_spont_exc),
                                                              "V1_Inh_L2/3" : self.dim_random_sample_events(small_spont_inh)}

        # Kohonen map
        results["Kohonen map"] = {"V1_Exc_L2/3" : self.kohonen_map(r["A_exc_bandpass"],r["or_map"])}
        tags = [t for t in tags if "t_res:" not in t]
        
        for name, vv in results.items():
            for sheet, value in vv.items():
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        stimulus_id=stimulus_id,
                        value=value,
                        value_units=qt.dimensionless,
                        value_name=name,
                        tags=tags,
                        sheet_name=sheet,
                        analysis_algorithm=self.__class__.__name__,
                    )
                )


class MulhollandSmithPlots(Plotting):

    def truncate_colormap(self, cmap, minval=0.0, maxval=1.0, n=-1):
        if n == -1: 
            n = cmap.N 
        new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list( 
             'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval), 
             cmap(np.linspace(minval, maxval, n))) 
        return new_cmap 
    
    def single_metric_plot(self,ax,exp,model,ylim,ylabel,has_legend=False):
        exp = np.array(exp)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_linewidth(1.5)
    
        ax.plot(np.ones_like(exp)*1,exp,'s',color='silver')
        ax.plot([1],exp.mean(),'o',color='k')
        ax.plot([1],model,'ro')
        
        if model < ylim[0]:
            ylim[0] = 0.9 * model
        if model > ylim[1]:
            ylim[1] = 1.1 * model
            
        ax.set_ylim(ylim[0],ylim[1])
        ax.set_xlim(0.5,1.5)
        if has_legend:
            ax.legend(["Experiment","Exp. mean","Model"],
                       bbox_to_anchor=(1.85, 1.05),
                       frameon=False)

        ax.set_ylabel(ylabel)
        ax.set_xticks([1],[""])
    
    def double_metric_plot(self,ax,exp_0,exp_1,model_0,model_1,ylim,ylabel,has_legend=False,x_ticks=["",""]):
        exp_0 = np.array(exp_0)
        exp_1 = np.array(exp_1)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_linewidth(1.5)
    
        ax.plot(np.ones_like(exp_0)*1,exp_0,'s',color='silver')
        ax.plot([1],exp_0.mean(),'o',color='k')
        ax.plot([1],model_0,'ro')
    
        ax.plot(np.ones_like(exp_1)*2,exp_1,'s',color='silver')
        ax.plot([2],exp_1.mean(),'o',color='k')
        ax.plot([2],model_1,'ro')
        
        if model_0 < ylim[0]:
            ylim[0] = 0.9 * model_0
        if model_1 < ylim[0]:
            ylim[0] = 0.9 * model_1
    
        if model_0 > ylim[1]:
            ylim[1] =1.1 * model_0
        if model_1 > ylim[1]:
            ylim[1] = 1.1 * model_1
            
        ax.set_ylim(ylim[0],ylim[1])
        ax.set_xlim(0.5,2.5)
        if has_legend:
            ax.legend(["Experiment","Exp. mean","Model"],
                       bbox_to_anchor=(1.85, 1.05),
                       frameon=False)
        ax.set_ylabel(ylabel)
        ax.set_xticks([1,2],x_ticks)

    def get_experimental_data(self):
        return {
            "Smith 2018": {
                "similarity": [0.382,0.437,0.562,0.457,0.368,0.382,0.288,0.475],
                "spatial scale of correlation": [0.83419,0.72982,1.08257,0.96128,1.09259],
                "dimensionality": [4,7,15.7,15.9,21.4],
                "mean eccentricity": [0.7508,0.68167,0.69373,0.60048,0.59405],
                "local correlation eccentricity_hist": {
                    "x": [0.02380952, 0.07142857, 0.11904762, 0.16666667, 0.21428571, 0.26190476, 0.30952381, 0.35714286, 0.4047619 , 0.45238095, 0.5, 0.54761905, 0.5952381 , 0.64285714, 0.69047619,0.73809524, 0.78571429, 0.83333333, 0.88095238, 0.92857143,0.97619048],
                    "y": [0, 0, 0, 0.00267881, 0.00401822,0.00750067, 0.01392982, 0.01607286, 0.02518082, 0.04125368,0.05464774, 0.07232789, 0.08143584, 0.10393785, 0.11197428,0.11251005, 0.12643986, 0.09375837, 0.06804179, 0.04553978,0.01875167],
                },
            },
            "Mulholland 2021": {
                "exc inh similarity": [0.533,0.207,0.419],
                "corr above 2 mm": {
                    "exc": [0.26698,0.3316,0.19743,0.30415,0.27338,0.33148,0.26845],
                    "inh": [0.28544,0.20432,0.26624,0.22759,0.37394,0.32446,0.28704],
                },
                "corr wavelength": {
                    "exc": [0.77539,0.76498,0.74042,0.85129,0.81111,0.95075,1.01449],
                    "inh": [0.67875,0.92874,0.74371,1.16734,0.96917,0.83597,0.67894],
                },
                "mean eccentricity": {
                    "exc": [0.7323,0.7117,0.7022,0.6933,0.6839,0.6579,0.6219],
                    "inh": [0.6455,0.6567,0.6579,0.6697,0.7211,0.7347,0.7754],
                },
                "dimensionality": {
                    # Dimensionality (mean of 100x random sample of 30 events) different from Smith 2018 (all events 1x)
                    "exc": [9.6303,11.3543,6.6423,10.6972,8.3869,8.9095,9.6195],
                    "inh": [12.6281,10.702,11.636,8.9744,8.7557,12.129,10.5414],
                },
            },
            "Mulholland 2024 may": {
                "exc opt similarity": [0.50349,0.56358,0.6047,0.55066,0.45385,0.37765],
                "spontaneous": {
                    "modularity": [0.113758,0.094554,0.103049,0.121333,0.134739,0.084584,0.136109],
                    "wavelength": {
                        "hist": {
                            "x": [0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55],
                            "y": [0.00661,0.10761,0.41724,0.273,0.10933,0.03516,0.02023,0.01135,0,0.01048,0.00524],
                        },
                        "mean": [0.752101,0.702946,0.765684,0.829392,0.832626,0.836509,0.896983],
                        "std": [0.0965972,0.138543,0.0851731,0.0878505,0.147288,0.274197,0.174062],
                    },
    
                },
                "fullfield opto": {
                    "modularity": [0.119471,0.136224,0.138085,0.089227,0.103886,0.061541,0.191605],
                    "wavelength": {
                        "hist": {
                            "x": [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75],
                            "y": [0.005945,0.008341,0.025811,0.194454,0.308177,0.251508,0.119363,0.04852,0.019643,0.01318,0.005614,0.008601,0.002685,0.003032,0.007663], 
                        },
                        "mean": [0.81685,0.816851,0.717572,0.763818,0.871507,0.874743,0.847903],
                        "std": [0.075143,0.187204,0.193154,0.211712,0.0649419,0.257613,0.133793],
                    },
                },
                "fullfield_timecourse": {
                    "x": [-0.9189999999999999, -0.793, -0.655, -0.502, -0.372, -0.23500000000000001, -0.097, 0.02500000000000001, 0.186, 0.29300000000000004, 0.45399999999999996, 0.622, 0.76, 0.9359999999999999, 0.989, 1.05, 1.119, 1.203, 1.28, 1.3410000000000002, 1.4020000000000001, 1.4560000000000002, 1.524, 1.586, 1.647, 1.7160000000000002, 1.83, 1.8920000000000001, 1.945, 2.006, 2.052, 2.106, 2.1750000000000003, 2.259, 2.335, 2.419, 2.496, 2.557, 2.618, 2.6870000000000003, 2.771, 2.878, 2.985, 3.092, 3.2070000000000003, 3.322, 3.482, 3.6510000000000002, 3.781, 3.926, 4.01, 4.129],
                    "y": [0.046,0.035,0.032,0.028,0.028,0.028,0.025,0.028,0.028,0.028,0.028,0.028,0.032,0.035,0.06,0.116,0.212,0.335,0.451,0.575,0.702,0.804,0.91,1.002,1.101,1.192,1.28,1.326,1.333,1.33,1.309,1.266,1.199,1.108,1.03,0.938,0.854,0.797,0.737,0.684,0.617,0.547,0.48,0.423,0.367,0.328,0.279,0.236,0.212,0.187,0.173,0.155],
                },
            },
            "Mulholland 2024 january": {
                "endo_inside_outside": {
                    "t": [-0.048600000000000004, -0.044500000000000005, -0.0442, -0.0396, -0.037500000000000006, 0.0063, 0.008099999999999996, 0.010699999999999994, 0.016399999999999998, 0.017100000000000004, 0.04859999999999999, 0.055999999999999994, 0.057300000000000004, 0.0762, 0.08039999999999999, 0.0957, 0.10640000000000001, 0.135, 0.14100000000000001, 0.1482, 0.14939999999999998, 0.19290000000000002, 0.19779999999999998, 0.2007, 0.2051, 0.24, 0.2409, 0.25420000000000004, 0.2692, 0.27840000000000004, 0.29460000000000003, 0.3115, 0.3254, 0.3428, 0.3484, 0.3773, 0.37820000000000004, 0.40700000000000003, 0.4096, 0.4338, 0.4464, 0.4684, 0.4915, 0.5, 0.5158999999999999, 0.5345, 0.5607, 0.5747, 0.5944999999999999, 0.6004999999999999, 0.6267999999999999, 0.6593, 0.6655, 0.6902999999999999, 0.6973999999999999, 0.7104999999999999, 0.7432, 0.7618999999999999, 0.7636999999999999, 0.7750999999999999, 0.8120999999999999, 0.8206, 0.8238, 0.8385999999999999, 0.8785999999999999, 0.8899999999999999, 0.8933, 0.8993, 0.9249999999999999, 0.9328, 0.9392999999999999, 0.9543999999999999, 0.9649999999999999, 0.9661, 0.9758, 0.9881, 0.9956, 0.9989999999999999, 1.0205, 1.0212999999999999, 1.0315999999999999, 1.0327, 1.044, 1.0591, 1.0671, 1.0722, 1.0778999999999999, 1.0918999999999999, 1.0996, 1.1040999999999999, 1.1194, 1.1256, 1.1286, 1.1403999999999999, 1.1482999999999999, 1.1672, 1.169, 1.1774, 1.1827999999999999, 1.2026, 1.2103, 1.2173, 1.2175, 1.2406, 1.2462, 1.2517, 1.2538, 1.2674999999999998, 1.2913999999999999, 1.2949, 1.2985, 1.3123, 1.3334, 1.3371, 1.3407, 1.3647, 1.3695, 1.3755, 1.3865, 1.3951, 1.4041, 1.4233, 1.4294, 1.4299, 1.4329, 1.4585, 1.4667999999999999, 1.4799, 1.4929999999999999, 1.5022, 1.51, 1.5164, 1.5214999999999999, 1.5433, 1.5436999999999999, 1.5487, 1.5677999999999999, 1.5742, 1.5802, 1.5855, 1.6074, 1.6140999999999999, 1.63, 1.6472, 1.6493, 1.666, 1.6729, 1.6809, 1.7068999999999999, 1.7130999999999998, 1.7221, 1.7402, 1.7527, 1.7558, 1.756, 1.7986, 1.7988, 1.7989, 1.8019, 1.8405, 1.8459999999999999, 1.8473, 1.857, 1.8626, 1.8833, 1.887, 1.894, 1.9022, 1.9087, 1.9303, 1.9329, 1.9339, 1.9396, 1.9555999999999998, 1.9587, 1.9669999999999999, 1.9732999999999998, 1.9795, 1.9915999999999998, 1.9963, 1.9965, 2.0128000000000004, 2.027, 2.0324, 2.0326, 2.0544000000000002, 2.0583, 2.0645000000000002, 2.0731, 2.084, 2.0965000000000003, 2.1032, 2.1083000000000003, 2.1124, 2.1334, 2.1467, 2.1481000000000003, 2.1522, 2.1603000000000003, 2.1697, 2.1823, 2.1853000000000002, 2.1947, 2.2153, 2.2225, 2.2229, 2.2267, 2.2451000000000003, 2.2598000000000003, 2.2609000000000004, 2.2727000000000004, 2.274, 2.3064, 2.309, 2.3113, 2.3331000000000004, 2.3401, 2.3487, 2.3586, 2.3804000000000003, 2.3846000000000003, 2.3898, 2.4013, 2.4214, 2.422, 2.4449, 2.4514, 2.4585000000000004, 2.4866, 2.4955000000000003, 2.5127, 2.5232, 2.5317000000000003, 2.5325, 2.5631000000000004, 2.5717000000000003, 2.5740000000000003, 2.5771, 2.6168, 2.6245000000000003, 2.6350000000000002, 2.6532, 2.6574, 2.6824000000000003, 2.6839000000000004, 2.7067, 2.7242, 2.7361, 2.7497000000000003, 2.7567000000000004, 2.7870000000000004, 2.8067, 2.8088, 2.8237, 2.8561, 2.8638000000000003, 2.8743000000000003, 2.9093, 2.9137, 2.9392, 2.9576000000000002, 2.972, 3.0243, 3.0317000000000003, 3.0349000000000004, 3.0392, 3.0956, 3.1011, 3.1495, 3.157, 3.1753, 3.1934, 3.2563, 3.2680000000000002, 3.2851000000000004, 3.3333000000000004, 3.3671, 3.3763, 3.3783000000000003, 3.4218, 3.4876, 3.4886000000000004, 3.4893, 3.4975, 3.5792, 3.5809, 3.5934000000000004, 3.5983, 3.6715, 3.6813000000000002, 3.6917, 3.6999, 3.7572, 3.7670000000000003, 3.7739000000000003, 3.7783, 3.841, 3.8537000000000003, 3.8554000000000004, 3.8625000000000003, 3.8696, 3.9064, 3.9284000000000003, 3.9321, 3.9444000000000004, 3.9507000000000003, 3.9828, 3.9864000000000006, 3.9880000000000004, 4.0016],
                    "inside": [0.05014,0.04995,0.04994,0.04974,0.04965,0.04964,0.04964,0.04964,0.04964,0.04973,0.04994,0.04999,0.04999,0.04982,0.04978,0.04964,0.04954,0.04965,0.04968,0.04971,0.04971,0.04922,0.04916,0.04899,0.04871,0.04655,0.04652,0.04615,0.04573,0.04547,0.04501,0.04433,0.04377,0.04307,0.04284,0.04186,0.04183,0.04085,0.04084,0.04069,0.04062,0.04049,0.04017,0.04005,0.03983,0.03958,0.03934,0.03921,0.03903,0.03903,0.03903,0.03902,0.03902,0.03907,0.03908,0.0391,0.03916,0.03918,0.03919,0.03934,0.03983,0.03995,0.03999,0.04033,0.04125,0.04151,0.04179,0.04229,0.04445,0.0451,0.04565,0.04869,0.05082,0.05103,0.05299,0.05545,0.05807,0.05925,0.06671,0.06698,0.07057,0.07093,0.07477,0.07986,0.08256,0.0843,0.08623,0.09109,0.0938,0.09537,0.1007,0.10287,0.10403,0.10871,0.11183,0.11925,0.12009,0.12391,0.12638,0.13545,0.13856,0.14136,0.14143,0.15073,0.153,0.15572,0.15677,0.16361,0.17556,0.1773,0.17904,0.18571,0.1959,0.19769,0.19944,0.21183,0.21434,0.21744,0.22337,0.22803,0.23291,0.24498,0.24881,0.24911,0.25076,0.26465,0.26918,0.27521,0.2812,0.28547,0.28956,0.29296,0.29567,0.30716,0.30732,0.30929,0.31678,0.31928,0.32161,0.32372,0.33397,0.33715,0.3446,0.35081,0.35157,0.35761,0.36008,0.36349,0.37456,0.37718,0.3806,0.38742,0.39214,0.39332,0.39337,0.41045,0.4105,0.41056,0.41177,0.42748,0.4293,0.42972,0.43292,0.43477,0.44044,0.44133,0.44298,0.44492,0.44646,0.44936,0.44971,0.44984,0.4506,0.45233,0.45267,0.45357,0.45414,0.4547,0.45582,0.45559,0.45558,0.45479,0.4541,0.45349,0.45346,0.45095,0.4505,0.4492,0.44743,0.44518,0.44253,0.4411,0.44003,0.43915,0.43311,0.42847,0.42798,0.42654,0.42385,0.42068,0.41673,0.4158,0.41285,0.40445,0.40149,0.40132,0.39972,0.39187,0.38666,0.38624,0.38207,0.3816,0.36976,0.36881,0.36796,0.35993,0.35731,0.35414,0.35052,0.34253,0.341,0.3391,0.33505,0.32793,0.32773,0.31989,0.31767,0.31523,0.30573,0.30272,0.29644,0.29263,0.28953,0.28921,0.28061,0.2782,0.27757,0.27669,0.26309,0.26104,0.2582,0.25332,0.25219,0.24556,0.24517,0.23914,0.23506,0.23231,0.22914,0.22752,0.21965,0.21455,0.2141,0.2109,0.20392,0.20261,0.20082,0.19485,0.1941,0.18973,0.18658,0.1841,0.17589,0.17473,0.17433,0.17378,0.16662,0.16598,0.16045,0.1596,0.15751,0.15544,0.14932,0.14819,0.14652,0.14242,0.13954,0.13877,0.13865,0.13613,0.13233,0.13227,0.13223,0.13179,0.12735,0.12726,0.12658,0.12631,0.12225,0.12171,0.12138,0.12111,0.11927,0.11895,0.11873,0.11859,0.11653,0.11611,0.11608,0.11596,0.11583,0.11519,0.11481,0.11475,0.11457,0.11448,0.11402,0.11397,0.11394,0.11375],
                    "outside": [0.03569,0.03569,0.03569,0.0357,0.0357,0.03578,0.03578,0.03579,0.03582,0.03582,0.03595,0.03588,0.03587,0.03568,0.03564,0.03549,0.03535,0.03495,0.03487,0.03477,0.03474,0.03377,0.03365,0.03359,0.03345,0.03235,0.03232,0.03226,0.03219,0.03214,0.0318,0.03144,0.03114,0.03102,0.03098,0.03078,0.03076,0.03016,0.0301,0.0296,0.02934,0.02889,0.02841,0.02824,0.02812,0.02798,0.02778,0.02784,0.02792,0.02794,0.02804,0.0279,0.02787,0.02776,0.02774,0.02769,0.02756,0.02749,0.0275,0.02761,0.02794,0.02802,0.02803,0.0281,0.02828,0.02837,0.02839,0.02844,0.02863,0.02932,0.02988,0.03121,0.03214,0.03223,0.03394,0.03608,0.03739,0.03799,0.04201,0.04215,0.04408,0.04428,0.04675,0.05002,0.05175,0.05302,0.05443,0.05787,0.05972,0.06079,0.06444,0.06604,0.0668,0.06986,0.0719,0.07629,0.07671,0.07865,0.07991,0.0849,0.08684,0.08839,0.08843,0.09359,0.09519,0.09674,0.09734,0.10123,0.10783,0.10879,0.10978,0.11393,0.12027,0.12139,0.12242,0.12927,0.13066,0.13237,0.13549,0.13795,0.14044,0.14577,0.14749,0.14763,0.14848,0.15567,0.15798,0.1616,0.16521,0.16788,0.17011,0.17196,0.17307,0.1778,0.17788,0.17897,0.18396,0.18563,0.18698,0.18821,0.19318,0.19472,0.19882,0.20327,0.2037,0.20714,0.20854,0.21019,0.21564,0.21693,0.21883,0.22245,0.22495,0.22558,0.22561,0.23419,0.23421,0.23425,0.23484,0.24295,0.24412,0.24438,0.24565,0.24639,0.2491,0.24959,0.25033,0.2512,0.25189,0.25417,0.25445,0.25453,0.255,0.25632,0.25657,0.25726,0.25778,0.25811,0.25877,0.25902,0.25903,0.25822,0.25751,0.25724,0.25723,0.25447,0.25397,0.25317,0.25206,0.25063,0.24901,0.24813,0.24743,0.24687,0.244,0.24218,0.24192,0.24114,0.23964,0.23788,0.23553,0.23497,0.2329,0.22837,0.22678,0.22669,0.22593,0.22219,0.21921,0.21894,0.21631,0.21602,0.20876,0.20827,0.20783,0.2037,0.20236,0.20073,0.19885,0.19457,0.19375,0.19273,0.19047,0.18634,0.18622,0.18153,0.18021,0.17887,0.17358,0.1719,0.16866,0.16669,0.16513,0.16496,0.15931,0.15798,0.15764,0.15716,0.15103,0.14985,0.14823,0.14538,0.14473,0.14079,0.14057,0.13745,0.13505,0.13343,0.13156,0.13069,0.12694,0.12451,0.12426,0.12258,0.11891,0.11804,0.11689,0.11303,0.11255,0.10975,0.10827,0.10711,0.10292,0.10232,0.10207,0.10172,0.09721,0.09676,0.09376,0.0933,0.09216,0.09124,0.08804,0.08745,0.08658,0.08413,0.08282,0.08246,0.08239,0.0807,0.07834,0.07831,0.07828,0.07799,0.07555,0.0755,0.07519,0.07507,0.07324,0.073,0.07274,0.07267,0.07219,0.07212,0.07207,0.07203,0.07157,0.07147,0.07146,0.07141,0.07137,0.07114,0.071,0.07098,0.07092,0.07089,0.07074,0.07072,0.07071,0.07065],
                },
            },
        }

    def values_from_queries(self,q):
        v = {}
        for k in q.keys():
            v[k] = q[k].get_analysis_result()
            assert len(v[k]) == 1, "Must need exactly 1 %s, got %d" % (k,len(v[k]))
            assert type(v[k][0]) == SingleValue
            v[k] = v[k][0].value
        return v

class Smith2018Mulholland2024Plot(MulhollandSmithPlots):   
    required_parameters = ParameterSet({})   

    def circ_dist(self, a, b):
            return np.pi/2 - abs(np.pi/2 - abs(a-b))

    def plot_hist_comparison(self,ax,e0,h0,e1,h1,title="",xlabel="",e1_center=False,ylim=[0,0.8]):
        e0, e1, h0, h1 = np.array(e0), np.array(e1), np.array(h0), np.array(h1), 
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_linewidth(1.5)
        ax.bar(e0,h0,width=e0[1]-e0[0],align='center',alpha=0.4,color='k') #edgecolor='black', color='none',lw=2)
        if not e1_center:
            e1 += (e1[1] - e1[0]) / 2
            e1 = e1[:-1]
        ax.bar(e1,h1,width=e1[1]-e1[0],align='center',alpha=0.4,color='r')#, edgecolor='red', color='none',lw=2)
        ax.legend(['Exp.','Model'],frameon=False,handlelength=0.8,loc='upper right')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability")
        ax.set_yticks([])
        mmin, mmax = min(e0.min(),e1.min()), max(e0.max(),e1.max())
        mmax = mmax + mmin
        mmin = 0
        ax.set_xlim(mmin,mmax)
        ax.set_xticks([mmin,mmin+(mmax-mmin)/2,mmax])
        ax.set_ylim(ylim[0],ylim[1])
        
    def subplot(self, subplotspec):

        dsv = queries.param_filter_query(self.datastore,analysis_algorithm="Smith_2018_Mulholland_2021_2024_spont_analyses")
        q = {
            "Orientation map similarity": queries.param_filter_query(self.datastore,value_name="orientation map similarity",sheet_name="V1_Exc_L2/3",st_name='InternalStimulus'),
            "Orientation map": queries.param_filter_query(self.datastore,value_name="orientation map"),
            "Excitatory correlation maps": queries.param_filter_query(self.datastore,analysis_algorithm="CorrelationMaps",sheet_name="V1_Exc_L2/3",st_name='InternalStimulus'),
            "Inhibitory correlation maps": queries.param_filter_query(self.datastore,analysis_algorithm="CorrelationMaps",sheet_name="V1_Inh_L2/3",st_name='InternalStimulus'),
            "Kohonen map": queries.param_filter_query(dsv,value_name="Kohonen map",sheet_name="V1_Exc_L2/3"),
            "Spatial scale of correlation": queries.param_filter_query(dsv,value_name="Spatial scale of correlation",sheet_name="V1_Exc_L2/3"),
            "Local correlation eccentricity": queries.param_filter_query(dsv,value_name="Local correlation eccentricity",sheet_name="V1_Exc_L2/3"),
            "Dimensionality": queries.param_filter_query(dsv,value_name="Dimensionality",sheet_name="V1_Exc_L2/3"),
            "Event activity wavelength": queries.param_filter_query(dsv,value_name="Event activity wavelength",sheet_name="V1_Exc_L2/3"),
            "Event activity modularity": queries.param_filter_query(dsv,value_name="Event activity modularity",sheet_name="V1_Exc_L2/3"),
        }
            
        d = self.get_experimental_data()["Smith 2018"]
        d_ = self.get_experimental_data()["Mulholland 2024 may"]["spontaneous"]
        plots = {}
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            10, 35, subplot_spec=subplotspec, hspace=0.3, wspace=0.2
        )
        v = self.values_from_queries(q)
        print(v["Dimensionality"])
        upper_w = 6
        # Orientation map
        ax = pylab.subplot(gs[0:3,0:upper_w-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ormap = pylab.imshow(v["Orientation map"],'hsv', interpolation='none')
        pylab.title("Orientation map",fontsize=10)
        #pylab.axis('equal')
        cbar = pylab.colorbar(ormap)
        cbar.set_label(label='Orientation preference', labelpad=5, fontsize=10)
        cbar.set_ticks([0,np.pi],labels=["0","$\\pi$"])

        cmap_idx = 8186
        cmap_x, cmap_y = cmap_idx % v["Excitatory correlation maps"].shape[-1], cmap_idx // v["Excitatory correlation maps"].shape[-1]
        # Exc correlation map
        ax = pylab.subplot(gs[0:3,1*upper_w:2*upper_w-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        cmap = pylab.imshow(v["Excitatory correlation maps"][cmap_idx],'bwr',vmin=-1,vmax=1)
        pylab.title("Excitatory\ncorrelation map",fontsize=10)
        ax.scatter(cmap_x,cmap_y,color='k',marker='x')
        cbar = pylab.colorbar(cmap)
        cbar.set_label(label='Correlation', labelpad=5, fontsize=10)
        cbar.set_ticks([-1,1],labels=["-1","1"])

        # Inh correlation map
        ax = pylab.subplot(gs[0:3,2*upper_w:3*upper_w-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        cmap = pylab.imshow(v["Inhibitory correlation maps"][cmap_idx],'bwr',vmin=-1,vmax=1)
        pylab.title("Inhibitory\ncorrelation map",fontsize=10)
        ax.scatter(cmap_x,cmap_y,color='k',marker='x')
        #pylab.axis('equal')
        cbar = pylab.colorbar(cmap)
        cbar.set_label(label='Correlation', labelpad=5, fontsize=10)
        cbar.set_ticks([-1,1],labels=["-1","1"])
        
        # Similarity map
        ax=pylab.subplot(gs[0:3, 3*upper_w:4*upper_w-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        inferno_t = self.truncate_colormap(plt.get_cmap("hot"), 0.5, 1) 
        smap = pylab.imshow(v["Orientation map similarity"].reshape(v["Orientation map"].shape), vmin=0, vmax=1,cmap=inferno_t)
        #pylab.axis("equal")
        pylab.title("Orientation map similarity\nmean=%.2f" % v["Orientation map similarity"].mean(),fontsize=10)
        cbar = pylab.colorbar(smap)
        cbar.set_label(label='Similarity', labelpad=5, fontsize=10)    
        cbar.set_ticks([0,1])
        
        # Kohonen map
        ax = pylab.subplot(gs[0:3,4*upper_w:5*upper_w-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        kohonenmap = pylab.imshow(v["Kohonen map"],'hsv', interpolation='none')
        estimation_error=np.nanmean(self.circ_dist(v["Orientation map"],v["Kohonen map"])) / np.pi * 180
        pylab.title("Kohonen map; error=%.2f°" % estimation_error,fontsize=10)
        cbar = pylab.colorbar(kohonenmap)
        cbar.set_label(label='Estimated orientation', labelpad=5, fontsize=10)
        cbar.set_ticks([0,np.pi],labels=["0","$\\pi$"])

        # Experiment comparison
        # Similarity
        self.single_metric_plot(pylab.subplot(gs[4:8,0:3]),d["similarity"],v["Orientation map similarity"].mean(),[0,1],"Exc $\\rightarrow$ or. map similarity",has_legend=False)
        self.single_metric_plot(pylab.subplot(gs[4:8,4:7]),d["spatial scale of correlation"],v["Spatial scale of correlation"],[0,1.2],"Spatial scale of correlation",has_legend=False)
        self.single_metric_plot(pylab.subplot(gs[4:8,8:11]),d["dimensionality"],v["Dimensionality"],[0,20],"Dimensionality",has_legend=False)
        lce_exc_bins = d["local correlation eccentricity_hist"]["x"]
        lce_exc_bins = np.hstack([lce_exc_bins,lce_exc_bins[-1]+lce_exc_bins[0]*2])
        lce_exc_bins -= lce_exc_bins[0]
        lce_exc_hist, _ = np.histogram(v["Local correlation eccentricity"],bins=lce_exc_bins)
        lce_exc_hist = lce_exc_hist.astype(float) / lce_exc_hist.sum()
        self.plot_hist_comparison(pylab.subplot(gs[4:8,12:15]),d["local correlation eccentricity_hist"]["x"],d["local correlation eccentricity_hist"]["y"],d["local correlation eccentricity_hist"]["x"],lce_exc_hist,title="",xlabel="Local correlation eccentricity",e1_center=True,ylim=[0,0.2])
        self.single_metric_plot(pylab.subplot(gs[4:8,17:20]),d["mean eccentricity"],v["Local correlation eccentricity"].mean(),[0,1],"Mean eccentricity",has_legend=False)
        self.single_metric_plot(pylab.subplot(gs[4:8,22:25]),d_["wavelength"]["mean"],v["Event activity wavelength"].mean(),[0,1.5],"Event activity wavelength",has_legend=False)
        self.single_metric_plot(pylab.subplot(gs[4:8,27:30]),d_["modularity"],v["Event activity modularity"].mean(),[0,0.2],"Modularity",has_legend=True)
        return plots

class Mulholland2021Plot(MulhollandSmithPlots):   
    required_parameters = ParameterSet({})
    
    def subplot(self, subplotspec):

        dsv = queries.param_filter_query(self.datastore,analysis_algorithm="Smith_2018_Mulholland_2021_2024_spont_analyses")
        q = {
            "Exc-Inh correlation map similarity": queries.param_filter_query(self.datastore,value_name="correlation map similarity"),
            "Exc local maxima distance correlation": queries.param_filter_query(dsv,value_name="Local maxima distance correlation",sheet_name="V1_Exc_L2/3"),
            "Inh local maxima distance correlation": queries.param_filter_query(dsv,value_name="Local maxima distance correlation",sheet_name="V1_Inh_L2/3"),
            "Exc correlation map wavelength": queries.param_filter_query(dsv,value_name="Correlation map wavelength",sheet_name="V1_Exc_L2/3"),
            "Inh correlation map wavelength": queries.param_filter_query(dsv,value_name="Correlation map wavelength",sheet_name="V1_Inh_L2/3"),
            "Exc local correlation eccentricity": queries.param_filter_query(dsv,value_name="Local correlation eccentricity",sheet_name="V1_Exc_L2/3"),
            "Inh local correlation eccentricity": queries.param_filter_query(dsv,value_name="Local correlation eccentricity",sheet_name="V1_Inh_L2/3"),
            "Exc dimensionality (random sampled events)": queries.param_filter_query(dsv,value_name="Dimensionality (random sampled events)",sheet_name="V1_Exc_L2/3"),
            "Inh dimensionality (random sampled events)": queries.param_filter_query(dsv,value_name="Dimensionality (random sampled events)",sheet_name="V1_Inh_L2/3"),
        }
        v = self.values_from_queries(q)
        d = self.get_experimental_data()["Mulholland 2021"]
        plots = {}
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 50, subplot_spec=subplotspec, hspace=0.3, wspace=0.2
        ) 
        self.single_metric_plot(pylab.subplot(gs[0:6]),d['exc inh similarity'],v["Exc-Inh correlation map similarity"].mean(),[0,1],"Inh $\\rightarrow$ Exc similarity")
        self.double_metric_plot(pylab.subplot(gs[10:16]),d['corr above 2 mm']['exc'],d['corr above 2 mm']['inh'],v["Exc local maxima distance correlation"][1,v["Exc local maxima distance correlation"][0,:] > 2].mean(),v["Inh local maxima distance correlation"][1,v["Inh local maxima distance correlation"][0,:] > 2].mean(),[0,0.5],"Correlation at maxima (>2 mm)",x_ticks=["Exc","Inh"])
        self.double_metric_plot(pylab.subplot(gs[20:26]),d['mean eccentricity']['exc'],d['mean eccentricity']['inh'],v["Exc local correlation eccentricity"].mean(),v["Inh local correlation eccentricity"].mean(),[0,1],"Mean local correlation eccentricity",x_ticks=["Exc","Inh"])
        self.double_metric_plot(pylab.subplot(gs[30:36]),d['corr wavelength']['exc'],d['corr wavelength']['inh'],v["Exc correlation map wavelength"],v["Inh correlation map wavelength"],[0.5,1.3],"Wavelength (mm)",x_ticks=["Exc","Inh"])
        self.double_metric_plot(pylab.subplot(gs[40:46]),d['dimensionality']['exc'],d['dimensionality']['inh'],v["Exc dimensionality (random sampled events)"].mean(),v["Inh dimensionality (random sampled events)"].mean(),[0,20],"Dimensionality",has_legend=True,x_ticks=["Exc","Inh"])
        return plots
