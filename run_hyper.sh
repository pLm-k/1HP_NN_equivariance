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

RUN_TAG="ecnn_pk_1000dp_hyperparam_${SLURM_JOB_ID:-manual}"

python main_hyperparam.py --dataset_raw dataset_square_3000dp_p_right_res5 \
    --device 'cuda:0'\
    --epochs 500 \
    --num_data_points 1000 \
    --destination "$RUN_TAG" \
    --always_load_default_lr_schedule \
    --equivariance_case ecnn \
    --crop \
    --inputs pk