#!/bin/bash
#SBATCH --partition=dell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:2
#SBATCH -J scinet
#SBATCH -o qslog/job-%j.log
#SBATCH -e qslog/job-%j.err
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
srun -u main.py --name qs --model qs --data ETTh2 --seq_len 48 --pred_len 24 --label_len 48 \
    --hidden_size 256 --n_heads 2 --e_layers 2 --encoder_attention "query_selector_0.85" --decoder_attention "full" \
    --d_layers 2 --batch_size 144 --embedding_size 64 --dropout 0.05 --features M --itr 3

srun -u main.py --name qs --model qs --data ETTh2 --seq_len 720 --pred_len 48 --label_len 48 \
    --hidden_size 128 --n_heads 2 --e_layers 3 --encoder_attention "query_selector_0.85" --decoder_attention "full" \
    --d_layers 3 --batch_size 32 --embedding_size 32 --dropout 0.15 --features M --itr 3

srun -u main.py --name qs --model qs --data ETTh2 --seq_len 336 --pred_len 168 --label_len 336 \
    --hidden_size 384 --n_heads 6 --e_layers 3 --encoder_attention "query_selector_0.8" --decoder_attention "full" \
    --d_layers 3 --batch_size 64 --embedding_size 96 --dropout 0 --features M --itr 3

srun -u main.py --name qs --model qs --data ETTh2 --seq_len 336 --pred_len 336 --label_len 336 \
    --hidden_size 96 --n_heads 6 --e_layers 3 --encoder_attention "query_selector_0.75" --decoder_attention "full" \
    --d_layers 3 --batch_size 32 --embedding_size 64 --dropout 0.15 --features M --itr 3

srun -u main.py --name qs --model qs --data ETTh2 --seq_len 336 --pred_len 720 --label_len 336 \
    --hidden_size 512 --n_heads 5 --e_layers 3 --encoder_attention "query_selector_0.90" --decoder_attention "full" \
    --d_layers 3 --batch_size 32 --embedding_size 64 --dropout 0.05 --features M --itr 3