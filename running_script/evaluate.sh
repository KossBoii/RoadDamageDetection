#!/bin/bash
#SBATCH --job-name=road_stress
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o ./slurm_log/output_%j.txt
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=lntruong@cpp.edu

eval "$(conda shell.bash hook)"
conda activate py3

echo "=========================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NODELIST: $SLURM_JOB_GPUS"
echo "=========================================="

if [ $# -eq 0 ]
    echo "No arguments supplied. Run evaluation script on all trained models under output folder"
	declare -a folderNames=()

	if [ $# == 0 ]
	then
		folderNames+=(`ls ./output/ -I ^06*$`)
	else
		for folder in "$@"
		do
			folderNames+=($folder)
		done
	fi

	for folder in "${folderNames[@]}"
	do
		if [[ ! -f "./output/$folder/model_final.pth" ]]
		then
			echo "model_final.pth file does not exist! Skip evaluating for this folder $folder"
		else
			echo "=========================================="
			srun python3 evaluate.py --model-name=$folder
			echo "=========================================="	
		fi
	done
else
	for folder in "$@"
	do
		if [[ ! -f "./output/$folder/model_final.pth" ]]
		then
			echo "model_final.pth file does not exist! Skip evaluating for this folder $folder"
		else
			echo "=========================================="
			srun python3 evaluate.py --model-name=$folder
			echo "=========================================="	
		fi
	done
fi