import numpy as np
from neurostim.analysis import quick_sim_setup

if str(snakemake.wildcards.target_n_locs) == '9999':
    # use all compartments of neuron according to impedance file:
    node_x_resistance = np.load(str(snakemake.input[0]))
    all_locs = node_x_resistance[:,0:2]
    np.save(str(snakemake.output),all_locs)

simcontrol = quick_sim_setup(
    cell_dict=dict(
        cellmodel=str(snakemake.wildcards.cell_id),
        ChR_soma_density=13e9,
        ChR_distribution='uniform'
    ),
    stimulator_dict=dict(
        diameter_um=200,
        NA=0.22
    )
)

def find_uni_locs(target, ph_tree):
    dx = 50
    results = {}
    cnt = 0
    while True:
        uni_locs = ph_tree.distributeLocsUniform(dx=dx)
        n_locs = len(uni_locs)
        results[n_locs]=dx
        cnt+=1
        if n_locs > target:
            dx = dx + 1
        elif n_locs < target:
            dx = dx -1 
        else:
            break
        if cnt > 100:
            tested_n_locs = np.array([key for key in results.keys()])
            closest_dx = results[tested_n_locs[np.argmin(np.abs(tested_n_locs-target))]]
            uni_locs = ph_tree.distributeLocsUniform(dx=closest_dx)
            break
    return uni_locs
        
def find_closest_number(numbers, target):
    closest_num = None
    smallest_diff = float('inf')  # Initialize with infinity
    
    for number in numbers:
        diff = abs(number - target)
        if diff < smallest_diff:
            smallest_diff = diff
            closest_num = number
    
    return closest_num

def get_seg_locs_from_locs(locs, sections):
    sec_xs = {}
    for sec in simcontrol.cell.sections:
        sec_xs[int(sec.name())]=[seg.x for seg in sec]
        
    seg_locs = []
    for locdict in locs:
        real_x = find_closest_number(sec_xs[locdict['node']], locdict['x'])
        seg_locs.append(dict(
            node=locdict['node'],
            x=real_x
        ))
    return seg_locs
    
ph_tree = simcontrol.cell.ph_tree
uni_locs = find_uni_locs(target=int(snakemake.wildcards.target_n_locs), ph_tree=ph_tree)
seg_uni_locs = get_seg_locs_from_locs(uni_locs, simcontrol.cell.sections)

np.save(str(snakemake.output),[[loc['node'], loc['x']] for loc in seg_uni_locs])
