#!/bin/bash
#SBATCH --partition=dell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:2
#SBATCH -J pred
#SBATCH -o zhylog/job-%j.log
#SBATCH -e zhylog/job-%j.err
# shellcheck disable=SC2046
# shellcheck disable=SC2006
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo "$SLURM_JOB_NODELIST"
cd ..
eval "$(conda shell.bash hook)"
source /home/LAB/anaconda3/bin/activate base
# conda --version
# which python
conda activate cpy
echo Python:
which python
export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export MKL_THREADING_LAYER=GNU
export CUDA_HOME=/usr/local/cuda-10.2
# sugon does not support infiniband
srun python -u main.py --fourrier --pred_len 24 --lambda_par 0.6 --A_lr 0.0002 --A_weight_decay 0 \
        --w_weight_decay 0.01 --fourier_divider 40 --temp 1 --sigmoid 1 --name param1 --data ETTh2 --data_path ETTh2.csv
srun python -u main.py --fourrier --pred_len 48 --lambda_par 0.6 --A_lr 0.0002 --A_weight_decay 0 \
        --w_weight_decay 0.001 --fourier_divider 40 --temp 1 --sigmoid 1 --name param1 --data ETTh2 --data_path ETTh2.csv
srun python -u main.py --fourrier --pred_len 168 --lambda_par 0.6 --A_lr 0.0002 --A_weight_decay 0 \
        --w_weight_decay 0.0008 --fourier_divider 40 --temp 5 --sigmoid 1 --name param2 --data ETTh2 --data_path ETTh2.csv