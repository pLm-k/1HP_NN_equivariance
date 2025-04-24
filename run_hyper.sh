#!/bin/bash
#SBATCH --job-name=1HP_NN_RUN_TRAIN
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=simcl1n2
#SBATCH --time=48:00:00

#module load cuda/12.2.2
source /import/sgs.scratch/miliczpl/cnn_env/bin/activate

python main_hyperparam.py --dataset_raw dataset_square_3000dp_p_rotate_res5 \
    --inputs pksi \
    --equivariance_case oriented_boxes \
    --device 'cuda:0'\
    --epochs 3000 \
    --destination '/import/sgs.scratch/miliczpl/models/cnn_3000_oriented_boxes' \