#!/bin/bash
#SBATCH -J "PSF"
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 1000


#export OPENBLAS_NUM_THREADS=4

module unload cuda
module load cuda/10.0.130
#module load cuda/9.0.176

echo "Starting bashscript3.sh"
echo "Starting Fiji script"

IMAGEJ_BINARY=$1
FIJI_MACRO_FILEPATH=$2
IMAGEJ_PLUGIN_PARAMS=$3
echo ${FIJI_MACRO_FILEPATH}
echo ${IMAGEJ_PLUGIN_PARAMS}

${IMAGEJ_BINARY} --headless -batch ${FIJI_MACRO_FILEPATH} ${IMAGEJ_PLUGIN_PARAMS}

echo "Finishing Fiji script"
echo "Finishing bashscript3.sh"
