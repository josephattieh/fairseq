#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=project_2007095
#SBATCH --partition=gpusmall
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60G
#SBATCH --output=/scratch/project_2007095/attieh/All_distillation_methods/logs/training/training_%j.log
#SBATCH --error=/scratch/project_2007095/attieh/All_distillation_methods/logs/training/training_%j.log

module load python-data/3.10
source /scratch/project_2007095/attieh/All_distillation_methods/myenv/bin/activate

# Define required variables
DIR="/scratch/project_2007095/attieh/All_distillation_methods"
FAIRSEQ="$DIR/fairseq/fairseq_cli"
CSV_FILE="$DIR/fairseq/scripts-joseph/training/archs.csv"
src=$1
trg=$2
model_name=$3

# Validate arguments
if [ -z "$src" ] || [ -z "$trg" ] || [ -z "$model_name" ]; then
    echo "Usage: $0 <src> <trg> <model_name>"
    exit 1
fi

DATASET=TATOEBA
MODEL_DIR="$DIR/models"
SPLIT_DIR=$DIR/data/$DATASET/split/
SPM_DIR=$DIR/data/$DATASET/$src-$trg/spm_model
SPM_MODEL=$SPM_DIR/spm_bpe_$src-$trg.model 
DATASET_DIR=$DIR/data/$DATASET/$src-$trg/spm_bpe_$src-$trg
tag="$DATASET/teacher/${src}-${trg}/${model_name}"

mkdir -p $MODEL_DIR/$tag
mkdir -p $MODEL_DIR/tensorboard_logs/$tag

set_model_params() {
    local model_name=$1
    local csv_file=$2

    # Use grep to find the row that matches the model_name
    local row=$(grep "^$model_name," "$csv_file")

    # Check if grep found a row
    if [ -z "$row" ]; then
        echo "Model $model_name not found in $csv_file"
        return 1
    fi

    # Split the row into variables using IFS and comma delimiter
    IFS=',' read -r model_name encoder_embed_dim encoder_ffn_embed_dim encoder_layers encoder_attention_heads encoder_normalize_before decoder_embed_dim decoder_ffn_embed_dim decoder_layers decoder_attention_heads decoder_normalize_before share_all_embeddings <<< "$row"



}
set_model_params $model_name $CSV_FILE

echo "Training model: $model_name"
share_all_embeddings_flag=""
encoder_normalize_flag=""
decoder_normalize_flag=""

    # Handle boolean flags for encoder and decoder normalization
if [ "$encoder_normalize_before" == "TRUE" ]; then
        encoder_normalize_flag="--encoder-normalize-before"
        decoder_normalize_flag="--decoder-normalize-before"        
fi
    
if [ "$share_all_embeddings" == "TRUE" ]; then
        share_all_embeddings_flag="--share-all-embeddings"
fi 


python3 -u $FAIRSEQ/train.py \
           $DATASET_DIR \
         --dataset-impl raw \
         --source-lang $1 \
         --target-lang $2 \
         --fp16 \
         --task translation --arch transformer \
         --encoder-embed-dim "$encoder_embed_dim" \
         --encoder-ffn-embed-dim "$encoder_ffn_embed_dim" \
         --encoder-layers "$encoder_layers" \
         --encoder-attention-heads "$encoder_attention_heads" \
         $encoder_normalize_flag \
         --decoder-embed-dim "$decoder_embed_dim" \
         --decoder-ffn-embed-dim "$decoder_ffn_embed_dim" \
         --decoder-layers "$decoder_layers" \
         --decoder-attention-heads "$decoder_attention_heads" \
         $decoder_normalize_flag \
         $share_all_embeddings_flag \
         --save-dir "$MODEL_DIR/$tag" \
         --tensorboard-logdir "$MODEL_DIR/tensorboard_logs/$tag" \
         --log-format simple \
         --log-interval 100 \
         --ddp-backend=legacy_ddp --log-interval 100 --log-format json \
         --max-tokens 4096 \
         --optimizer adam  --adam-betas '(0.9, 0.98)' \
         --lr 7e-4  --lr-scheduler inverse_sqrt \
         --warmup-updates 4000 --warmup-init-lr 1e-07 --update-freq 2 \
         --clip-norm 5.0 --dropout 0.1  \
         --weight-decay 0.0001 \
         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
         --max-epoch 100  --patience 5 \
         --seed 3921 \
         --keep-interval-updates 40 \
         --eval-bleu \
         --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
         --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
         --eval-bleu-remove-bpe sentencepiece \
         --bpe sentencepiece \
         --sentencepiece-model $SPM_MODEL    