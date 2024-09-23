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

DATASET=TATOEBA
src=$1
trg=$2
DATA_DIR=$DIR/data/$DATASET/$src-$trg/raw
SPM_DIR=$DIR/data/$DATASET/$src-$trg/spm_model
SPM_MODEL=$SPM_DIR/spm_bpe_$src-$trg.model 
OUTPUT_DIR=$DIR/data/$DATASET/$src-$trg/spm_bpe_$src-$trg
mkdir -p $OUTPUT_DIR


# Define an array of filenames
declare -A files=(
    ["train.src"]="train.$src-$trg.$src"
    ["train.trg"]="train.$src-$trg.$trg"
    ["dev.src"]="valid.$src-$trg.$src"
    ["dev.trg"]="valid.$src-$trg.$trg"
    ["test.src"]="test.$src-$trg.$src"
    ["test.trg"]="test.$src-$trg.$trg"
)


for key in "${!files[@]}"; do
    echo "Tokenizing ... $key"
    $SPM/spm_encode --model="$SPM_MODEL" \
        --output_format=piece < $DATA_DIR/$key >  $OUTPUT_DIR/${files[$key]}
    echo "Done tokenizing ... $key"

done

 
#source /scratch/project_2007095/attieh/All_distillation_methods/fairseq/scripts-joseph/spm_models/tokenize_spm.sh \
#    eng fao