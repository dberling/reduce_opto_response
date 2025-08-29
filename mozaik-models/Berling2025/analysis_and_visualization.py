from msa_analysis_plotting import *

def analysis(data_store):
    t_res, s_res, array_width = 5, 50, 4000
    RecordingArrayTimecourse(queries.param_filter_query(data_store,sheet_name="V1_Exc_L2/3"),
        ParameterSet(
            {
                "s_res": s_res,
                "t_res": t_res,
                "array_width": array_width,
                "electrode_radius": 50,
            }
        ),
    ).analyse()
    data_store.save()
