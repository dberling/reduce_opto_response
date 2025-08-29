#!/bin/bash

#source activate redupsin3
#snakemake --snakefile snake_workflow/Snakefile_adex_preparation --configfile snake_workflow/newconfig_L23_1_d50_NA01.yml --latency-wait 90 --rerun-incomplete --cores 1
#conda deactivate

#source activate snake_nestml_pip
#snakemake --snakefile snake_workflow/Snakefile_adex --configfile snake_workflow/newconfig_L23_1_d50_NA01.yml --latency-wait 90 --rerun-incomplete --cores 1
#conda deactivate

source activate redupsin3
snakemake --snakefile snake_workflow/Snakefile_adex_preparation --configfile snake_workflow/newconfig_L23_1_d100_NA039.yml --latency-wait 90 --rerun-incomplete --cores 1
conda deactivate

source activate snake_nestml_pip
snakemake --snakefile snake_workflow/Snakefile_adex --configfile snake_workflow/newconfig_L23_1_d100_NA039.yml --latency-wait 90 --rerun-incomplete --cores 1
conda deactivate

source activate redupsin3
snakemake --snakefile snake_workflow/Snakefile_adex_preparation --configfile snake_workflow/newconfig_L23_1_d100_NA09.yml --latency-wait 90 --rerun-incomplete --cores 1
conda deactivate

source activate snake_nestml_pip
snakemake --snakefile snake_workflow/Snakefile_adex --configfile snake_workflow/newconfig_L23_1_d100_NA09.yml --latency-wait 90 --rerun-incomplete --cores 1
conda deactivate
