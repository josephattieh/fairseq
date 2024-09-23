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

DICT=$SPM_DIR/spm_bpe_$src-$trg.vocab
OUTPUT_DIR=$DIR/data/$DATASET/$src-$trg/spm_bpe_$src-$trg

cut -f1 $DICT | \
         tail -n +5 | \
          sed "s/$/ 100/g" > $OUTPUT_DIR/dict.$src.txt

cp $OUTPUT_DIR/dict.$src.txt $OUTPUT_DIR/dict.$trg.txt
cp $OUTPUT_DIR/dict.$src.txt $OUTPUT_DIR/dict.txt