import pandas as pd

data = pd.concat([pd.read_csv(fname, index_col=0) for fname in list(snakemake.input)])

data.to_csv(str(snakemake.output))
