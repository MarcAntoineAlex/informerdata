#!/bin/bash
#SBATCH --partition=dell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:2
#SBATCH -J scinet
#SBATCH -o scilog/job-%j.log
#SBATCH -e scilog/job-%j.err
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
conda activate gch
echo Python:
which python
export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export MKL_THREADING_LAYER=GNU
export CUDA_HOME=/usr/local/cuda-10.2
# sugon does not support infiniband
srun python main.py --data ETTh1 --features M  --seq_len 48 --label_len 24 --pred_len 24 --hidden-size 4 --stacks 1 \
--name scinet --levels 3 --lr 3e-3 --batch_size 8 --dropout 0.5 --model SCINet --train_eposhs 100 --patience 5
srun python main.py --data ETTh1 --features M  --seq_len 96 --label_len 48 --pred_len 48 --hidden-size 4 --stacks 1 \
--name scinet --levels 3 --lr 0.009 --batch_size 16 --dropout 0.25 --model SCINet --train_eposhs 100 --patience 5
srun python main.py --data ETTh1 --features M  --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 4 --stacks 1 \
--name scinet --model SCINet --train_eposhs 100 --patience 5 --levels 3 --lr 5e-4 --batch_size 32 --dropout 0.5
srun python main.py --data ETTh1 --features M  --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1 --stacks 1 \
--name scinet --model SCINet --train_eposhs 100 --patience 5 --levels 4 --lr 1e-4 --batch_size 512 --dropout 0.5
srun python main.py --data ETTh1 --features M  --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 1 --stacks 1 \
--name scinet --model SCINet --train_eposhs 100 --patience 5 --levels 5 --lr 5e-5 --batch_size 256 --dropout 0.5

srun python main.py --data ETTh1 --features M  --seq_len 96 --label_len 48 --pred_len 24 --hidden-size 4 --stacks 1 \
--name scinet --model SCINet --train_eposhs 100 --patience 5 --levels 3 --lr 3e-3 --batch_size 8 --dropout 0.5
srun python main.py --data ETTh1 --features M  --seq_len 96 --label_len 48 --pred_len 48 --hidden-size 4 --stacks 1 \
--name scinet --model SCINet --train_eposhs 100 -patience 5 --levels 3 --lr 0.009 --batch_size 16 --dropout 0.25
srun python main.py --data ETTh1 --features M  --seq_len 96 --label_len 48 --pred_len 168 --hidden-size 4 --stacks 1 \
--name scinet --model SCINet --train_eposhs 100 --patience 5 --levels 3 --lr 5e-4 --batch_size 32 --dropout 0.5
srun python main.py --data ETTh1 --features M  --seq_len 96 --label_len 48 --pred_len 336 --hidden-size 1 --stacks 1 \
--name scinet --model SCINet --train_eposhs 100 --patience 5 --levels 4 --lr 1e-4 --batch_size 512 --dropout 0.5
srun python main.py --data ETTh1 --features M  --seq_len 96 --label_len 48 --pred_len 720 --hidden-size 1 --stacks 1 \
--name scinet --model SCINet --train_eposhs 100 --patience 5 --levels 5 --lr 5e-5 --batch_size 256 --dropout 0.5