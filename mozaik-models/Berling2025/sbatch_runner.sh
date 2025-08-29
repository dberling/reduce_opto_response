#!/bin/bash

#SBATCH -c 64
#SBATCH -w w14
#SBATCH -J Analysis
#SBATCH --hint=nomultithread
#SBATCH --mem=1400gb

#source /home/rozsa/virt_env/mozaik/bin/activate
source /home/rozsa/virt_env/mozaik_morphology/bin/activate

srun python recordarray_analysis_runner.py
