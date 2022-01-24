#!/bin/bash
#SBATCH --partition=dell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:2
#SBATCH -J abla_t
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
srun python -u main.py --fourrier --pred_len 168 --do_predict --lambda_par 0.6 --A_lr 0.0002 --A_weight_decay 0 \
        --w_weight_decay 0.0008 --fourier_divider 40 --temp 5 --sigmoid 1 --name ablation --data ETTh1 --itr 3 \
        --trigger --student_head 7

srun python -u main.py --fourrier --pred_len 168 --do_predict --lambda_par 0.6 --A_lr 0.0002 --A_weight_decay 0 \
        --w_weight_decay 0.0008 --fourier_divider 40 --temp 5 --sigmoid 1 --name ablation --data ETTh1 --itr 3 \
        --trigger --student_head 6

srun python -u main.py --fourrier --pred_len 168 --do_predict --lambda_par 0.6 --A_lr 0.0002 --A_weight_decay 0 \
        --w_weight_decay 0.0008 --fourier_divider 40 --temp 5 --sigmoid 1 --name ablation --data ETTh1 --itr 3 \
        --trigger --student_head 5

srun python -u main.py --fourrier --pred_len 168 --do_predict --lambda_par 0.6 --A_lr 0.0002 --A_weight_decay 0 \
        --w_weight_decay 0.0008 --fourier_divider 40 --temp 5 --sigmoid 1 --name ablation --data ETTh1 --itr 3 \
        --trigger --student_head 4

srun python -u main.py --fourrier --pred_len 168 --do_predict --lambda_par 0.6 --A_lr 0.0002 --A_weight_decay 0 \
        --w_weight_decay 0.0008 --fourier_divider 40 --temp 5 --sigmoid 1 --name ablation --data ETTh1 --itr 3 \
        --trigger --student_head 3

srun python -u main.py --fourrier --pred_len 168 --do_predict --lambda_par 0.6 --A_lr 0.0002 --A_weight_decay 0 \
        --w_weight_decay 0.0008 --fourier_divider 40 --temp 5 --sigmoid 1 --name ablation --data ETTh1 --itr 3 \
        --trigger --student_head 2

srun python -u main.py --fourrier --pred_len 168 --do_predict --lambda_par 0.6 --A_lr 0.0002 --A_weight_decay 0 \
        --w_weight_decay 0.0008 --fourier_divider 40 --temp 5 --sigmoid 1 --name ablation --data ETTh1 --itr 3 \
        --trigger --student_head 1