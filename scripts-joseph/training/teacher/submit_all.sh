# Define the array of model names
model_names=(
    "tiny_v1"
    "tiny_v2"
    "tiny_v3"
    "tiny_v4"
    "tiny_v5"
    "tiny_v6"
    "tiny_v7"
    "small_v1"
    "small_v2"
    "small_v3"
    "small_v4"
    "base_v1"
    "base_v2"
    "big_v1"
    "big_v2"
    "big_v3"
    "large_v1"
)
src=$1
trg=$2
# Iterate over the array
for model_name in "${model_names[@]}"; do
    echo "Lunching model training: $model_name"
    sbatch /scratch/project_2007095/attieh/All_distillation_methods/fairseq/scripts-joseph/training/teacher/submit.sh \
        $src $trg $model_name
done