# get nest 3.6

...

# get nest requirements with conda (mamba)

srun conda env create --name redupsin -f nest-simulator-3.6/environment.yml

# compile nest

in the conda environment:
cd nest-simulator-3.6
srun cmake ./
srun make
srun make install
srun make installcheck

# conda install snakemake, add channels bioconda (snakemake) & conda-forge (eido dependency)

* If you only want to run snakemake locally:

conda install -c bioconda -c conda-forge snakemake

* If you want to run snakemake via a slurm cluster:

conda install -c bioconda -c conda-forge snakemake snakemake-executor-plugin-cluster-generic

# pip-install packages:

## specific NestML:

download from here: https://github.com/LeanderEwert/nestml/tree/origin/LeanderEwert_vectorization

git checkout LeanderEwert_vectorization

pip install ./nestml

## specific NEAT:

need to fork and add my implementation of neatmodel
#download from here https://github.com/WillemWybo/NEAT-2/tree/enh/nestml-channel-generation

git checkout nestml-channel-generation

pip install ./neat

## neurostim package:

git clone git@github.com:dberling/simneurostim.git

pip install ./simneurostim/base-neurostim

## NEAST-BBP-models:

need to fork and implement my absolute paths change
git clone git@github.com:INM-6/NEAST_models.git

git checkout model-pipeline

## compile NEURON mod files:

neatmodels install -p NEAST_models/BBP/bbpchannels.py --neuronresource simneurostim/model/mod/optostimmods/

# Execute workflow

Run submit_snake.sh
