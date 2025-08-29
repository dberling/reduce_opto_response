# -*- coding: utf-8 -*-
import sys
from mozaik.meta_workflow.parameter_search import (
    CombinationParameterSearch,
    SlurmSequentialBackend,
)
import numpy
import time

# Morphology
slurm_options = ["-J optog", "--exclude=w[1-12,16-17]", "--mem=500gb", "--hint=nomultithread"]
# Non-morphology
#slurm_options = ["-J optog", "--exclude=w[9,11-17]", "--mem=60gb", "--hint=nomultithread"]
#slurm_options = ["-J optog", "--hint=nomultithread"]
# Testing
#slurm_options = ["-J optog", "--mem=30gb", "--hint=nomultithread"]

CombinationParameterSearch(
    SlurmSequentialBackend(
        num_threads=21,
        #num_threads=16,
        num_mpi=1,
        path_to_mozaik_env="/home/rozsa/virt_env/mozaik_morphology/bin/activate",
        slurm_options=slurm_options,
    ),
    {
        "trial": [0,100,200,300,400,500],
        #"trial": [0,1,2,3,4,5],
        #"trial": [0,1,2,3,4],
        #"trial": [0,1,2],
        # No connections
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.base_weight": [0],
        "sheets.l4_cortex_exc.L4ExcL4InhConnection.base_weight": [0],
        "sheets.l4_cortex_inh.L4InhL4ExcConnection.base_weight": [0],
        "sheets.l4_cortex_inh.L4InhL4InhConnection.base_weight": [0],
        "sheets.l4_cortex_exc.AfferentConnection.base_weight": [0],
        "sheets.l4_cortex_inh.AfferentConnection.base_weight": [0],
        "sheets.l23_cortex_exc.L4ExcL23ExcConnection.base_weight": [0],
        "sheets.l23_cortex_inh.L4ExcL23InhConnection.base_weight": [0],
        "sheets.l23_cortex_exc.L23ExcL23ExcConnection.base_weight": [0],
        "sheets.l23_cortex_exc.L23ExcL23InhConnection.base_weight": [0],
        "store_stimuli": [True],
    },
).run_parameter_search()
