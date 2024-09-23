#!/bin/bash

# Path to your CSV file
CSV_FILE="/scratch/project_2007095/attieh/All_distillation_methods/model_archs/archs.csv"

COMMON_PARAMS=(
    --source-lang de
    --target-lang en
    --dataset-impl raw
    --fp16
    --arch transformer
    --max-tokens 4096
    --optimizer adam
    --adam-betas '(0.9, 0.98)'
    --lr 7e-4
    --lr-scheduler inverse_sqrt
    --warmup-updates 4000
    --warmup-init-lr 1e-07
    --clip-norm 5.0
    --dropout 0.1
    --weight-decay 0.0001
    --criterion label_smoothed_cross_entropy
    --label-smoothing 0.1
    --max-epoch 100 
    --patience 5
    --seed 3921
    --keep-interval-updates 40
    --eval-bleu
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' 
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
    --eval-bleu-detok space  --eval-bleu-remove-bpe sentencepiece 
    --best-checkpoint-metric bleu 
    --maximize-best-checkpoint-metric 
    --update-freq 2
)


model_name="tiny_v1"
paste -d= \
  <(cat $CSV_FILE | dos2unix | grep "^model_name"  | tr ',' '\n')  \
  <(cat $CSV_FILE | dos2unix | grep "^$model_name" | tr ',' '\n')  | \
  while IFS='=' read -r var value; do
    export "$var=$value"
  done

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
        share_all_embeddings_flag="--share_all_embeddings"
fi 

# Construct the fairseq-train command
fairseq-train "$DATA_DIR" \
        "${COMMON_PARAMS[@]}" \
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
        --save-dir "checkpoints/$model_name" \
        --tensorboard-logdir "tensorboard_logs/$model_name" \
        --log-format simple \
        --log-interval 100 \
        --ddp-backend=legacy_ddp --log-interval 100 --log-format json

