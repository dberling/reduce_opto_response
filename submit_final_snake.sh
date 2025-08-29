#!/bin/bash


#snakemake --jobs 32 --snakefile snake_workflow/Snakefile --configfile snake_workflow/newconfig_L23_1_d50_NA01.yml --executor cluster-generic --cluster-generic-submit-cmd 'sbatch' --latency-wait 90 --rerun-incomplete
#snakemake --jobs 32 --snakefile snake_workflow/Snakefile --configfile snake_workflow/newconfig_L23_1_d100_NA039.yml --executor cluster-generic --cluster-generic-submit-cmd 'sbatch' --latency-wait 90 --rerun-incomplete
#snakemake --jobs 32 --snakefile snake_workflow/Snakefile --configfile snake_workflow/newconfig_L23_1_d100_NA09.yml --executor cluster-generic --cluster-generic-submit-cmd 'sbatch' --latency-wait 90 --rerun-incomplete
snakemake --jobs 32 --snakefile snake_workflow/Snakefile --configfile snake_workflow/newconfig_L23_3_d100_NA039.yml --executor cluster-generic --cluster-generic-submit-cmd 'sbatch' --latency-wait 90 --rerun-incomplete
snakemake --jobs 32 --snakefile snake_workflow/Snakefile --configfile snake_workflow/newconfig_L23_3_d100_NA09.yml --executor cluster-generic --cluster-generic-submit-cmd 'sbatch' --latency-wait 90 --rerun-incomplete
