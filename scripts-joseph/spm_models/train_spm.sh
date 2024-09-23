#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=project_2007095
#SBATCH --partition=medium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=230GB
#SBATCH --output=/scratch/project_2007095/attieh/All_distillation_methods/logs/spm_model/spm_model_%j.log
#SBATCH --error=/scratch/project_2007095/attieh/All_distillation_methods/logs/spm_model/spm_model_%j.log

DIR=/scratch/project_2007095/attieh/All_distillation_methods
export SPM=$DIR/vcpkg/packages/sentencepiece_x64-linux/tools/sentencepiece
export FAIRSEQ=$DIR/fairseq/fairseq_cli
#process train, valid and test splits
DATASET=TATOEBA
src=$1
trg=$2
DATA_DIR=$DIR/data/$DATASET/$src-$trg/raw
SPM_DIR=$DIR/data/$DATASET/$src-$trg/spm_model

mkdir -p $SPM_DIR
$SPM/spm_train \
    --input="$DATA_DIR/train.src,$DATA_DIR/train.trg,$DATA_DIR/dev.src,$DATA_DIR/dev.trg" \
    --model_prefix=$SPM_DIR/spm_bpe_$src-$trg \
    --vocab_size=32000 \
    --character_coverage=1.0 \
    --num_threads=8 \
    --model_type=bpe \
    --input_sentence_size=10000000 \
    --shuffle_input_sentence=true \
    --bos_id=0 --pad_id=1 --eos_id=2 --unk_id=3 


#source  /scratch/project_2007095/attieh/All_distillation_methods/fairseq/scripts-joseph/spm_models/train_spm.sh \
#     eng fao