#!/bin/bash
#SBATCH --job-name=llama-instruct-cpo
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=23:55:00
#SBATCH --partition=pli-c
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

BETA=$1
LR=$2
NUM_CLUSTERS=${3:-8}
CLUSTER_MAP=${4:-null}           # path to sidecar JSON, or `null` for K=1 smoke test
EMA_TAU=${5:-null}               # e.g. 0.001; null for frozen reference

find_free_port() {
    local port
    while true; do
        port=$(shuf -i 29500-29510 -n 1)
        if ! netstat -tuln | grep -q ":$port "; then
            echo "$port"
            break
        fi
    done
}

init_env() {
    module load anaconda3/2024.2
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate halos

    echo "Running on node: $(hostname)"
    echo "Machine Rank: $SLURM_PROCID"

    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=$(find_free_port | tr -d '\n')
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1

    echo "Master node: $MASTER_ADDR"
    echo "Number of nodes: $SLURM_JOB_NUM_NODES"
    echo "GPUs per node: $SLURM_GPUS_PER_NODE"
}

export -f find_free_port
export -f init_env

srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "
init_env
export MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export EXP_NAME=llama3-8B-instruct-cpo-K${NUM_CLUSTERS}-${BETA}-${LR}
export CKPT=/scratch/gpfs/ke7953/models/\$EXP_NAME/FINAL

# Build the EMA flag conditionally
EMA_FLAG=''
if [ \"${EMA_TAU}\" != 'null' ]; then
    EMA_FLAG=\"++loss.sync_reference=true ++loss.ema_tau=${EMA_TAU}\"
fi

# Build the cluster_map flag conditionally
CLUSTER_FLAG=''
if [ \"${CLUSTER_MAP}\" != 'null' ]; then
    CLUSTER_FLAG=\"++loss.cluster_map_path=${CLUSTER_MAP}\"
fi

accelerate launch \
    --config_file accelerate_config/fsdp_4gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    launch.py loss=cpo model=llama train_datasets=[ultrafeedback_armorm] test_datasets=[ultrafeedback_armorm] exp_name=\$EXP_NAME \
    ++cache_dir=/scratch/gpfs/ke7953/models \
    ++model.name_or_path=\$MODEL_PATH \
    ++lr=${LR} \
    ++loss.beta=${BETA} ++loss.desirable_weight=1.1 \
    ++loss.num_clusters=${NUM_CLUSTERS} \
    \$CLUSTER_FLAG \$EMA_FLAG \
    ++humanline=false ++n_examples=20_000 \
    ++model.batch_size=64 ++model.eval_batch_size=64

python -m train.sample \$CKPT --gpu_count 2 --output_file outputs/\$EXP_NAME.json
"
