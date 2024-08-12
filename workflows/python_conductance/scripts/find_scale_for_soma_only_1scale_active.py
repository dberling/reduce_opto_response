import numpy as np

cond_scale = []
for filepath in snakemake.input:
    # conductance across compartments:
    conds = np.load(filepath)
    cond_soma = conds[:,0] # only soma
    cond_sum = conds.sum(axis=1) # sum across compartments
    cond_scale.append(cond_sum.sum() / cond_soma.sum())

mean = np.mean(cond_scale)

np.save(str(snakemake.output), np.array(mean))
