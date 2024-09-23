#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=project_2007095
#SBATCH --partition=gpusmall
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60G
#SBATCH --output=/scratch/project_2007095/attieh/All_distillation_methods/logs/distill.txt
#SBATCH --error=/scratch/project_2007095/attieh/All_distillation_methods/logs/distill.txt

export CUDA_VISIBLE_DEVICES=0,1,2,3

module load python-data/3.10
source /scratch/project_2007095/attieh/All_distillation_methods/myenv/bin/activate
DIR=/scratch/project_2007095/attieh/All_distillation_methods
export SPM=$DIR/vcpkg/packages/sentencepiece_x64-linux/tools/sentencepiece
export FAIRSEQ=$DIR/fairseq/fairseq_cli
#process train, valid and test splits
DATASET=TED2020
SRC_DATA=TED2020.de-en.de
TGT_DATA=TED2020.de-en.en
DATA_DIR=$DIR/data/$DATASET/raw/
SPM_DIR=$DIR/data/$DATASET/spm/
SPLIT_DIR=$DIR/data/$DATASET/split/


output_dir=/scratch/project_2007095/attieh/All_distillation_methods/models/TED2020_distilled_seq_tiny_v1
data_dir=$SPLIT_DIR
distil_strategy=batch_level
disitl_rate=0.5
queue_size=30000



python3 -u $FAIRSEQ/generate.py \
    $DIR/data/$DATASET/infer_train/  \
    --source-lang de \
    --target-lang en \
    --dataset-impl raw \
    --path $DIR/models/TED2020/checkpoint_best.pt \
    --beam 5 \
    --batch-size 64  \
    --remove-bpe \
    --results-path "$DIR/results/${DATASET}_inference/" 

#move train set to a different directory, name test and infer
#extract s and pred from output
source  $DIR/misc/extract_output.sh     /scratch/project_2007095/attieh/All_distillation_methods/results/TED2020_inference/generate-test.txt

# move to dataset and use to SEQ-DISTILL
#train set is distillation set, same valid and test sets

python3 -u $FAIRSEQ/train.py \
     $DIR/data/$DATASET/infer_train/ \
    --task translation --arch transformer \
    --encoder-embed-dim 32 \
    --encoder-ffn-embed-dim 32 \
    --encoder-layers 1 \
    --encoder-attention-heads 2 \
    --decoder-embed-dim 32 \
    --decoder-ffn-embed-dim 32 \
    --decoder-layers 1 \
    --decoder-attention-heads 2 \
    --attention-dropout 0.1 --activation-dropout 0.1 --dropout 0.1  \
    --source-lang de \
    --target-lang en \
    --dataset-impl raw \
    --fp16 \
    --label-smoothing 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 5.0 \
    --lr 7e-4 --lr-scheduler inverse_sqrt \
    --save-dir $output_dir \
    --max-tokens 8000 \
    --max-epoch 100 \
    --patience 5 \
    --seed 3921 \
    --keep-interval-updates 40 \
    --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy  \
    --max-tokens 4096 \
    --ddp-backend=legacy_ddp --log-interval 100 --log-format json \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-bleu-detok space  --eval-bleu-remove-bpe sentencepiece \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --update-freq 2
        #--distil-strategy $distil_strategy --distil-rate $disitl_rate \
    #--difficult-queue-size $queue_size
    # --share-all-embeddings
       # --eval-bleu-print-samples \
python3 -u $FAIRSEQ/generate.py \
     $SPLIT_DIR \
    --dataset-impl raw \
    --source-lang de \
    --target-lang en \
    --path $DIR/models/${DATASET}_distilled_seq/checkpoint_best.pt \
    --beam 5 \
    --batch-size 32 \
    --remove-bpe \
    --results-path "$DIR/results/${DATASET}_distilled_seq/" \
    --max-len-b 200