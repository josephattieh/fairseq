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
input_file=$3
output_file=$4
SPM_DIR=$DIR/data/$DATASET/$src-$trg/spm_model
SPM_MODEL=$SPM_DIR/spm_bpe_$src-$trg.model 

$SPM/spm_decode --model="$SPM_MODEL" \
     --input_format=piece < $3 > $4

# source /scratch/project_2007095/attieh/All_distillation_methods/fairseq/scripts-joseph/spm_models/detokenize_spm.sh  \
#     eng fao \
#     /scratch/project_2007095/attieh/All_distillation_methods/data/TATOEBA/eng-fao/spm_bpe_eng-fao/test.eng-fao.eng \
#     /scratch/project_2007095/attieh/All_distillation_methods/data/TATOEBA/eng-fao/spm_bpe_eng-fao/test.eng-fao.eng.detok
 



