#!/bin/bash
#SBATCH --job-name=1HP_NN_RUN_TRAIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=simcl1n2
#SBATCH --time=48:00:00

#module load cuda/12.2.2
#source /import/sgs.scratch/miliczpl/cnn_env/bin/activate

python main.py --dataset_raw 1dp \
    --inputs pksi \
    --device 'cuda:0' \
    --epochs 5000 \
    --equivariance_case ecnn_cont \
    #--destination '/import/sgs.scratch/miliczpl/models/cont/ecnn_m-f4_f64_1000' \