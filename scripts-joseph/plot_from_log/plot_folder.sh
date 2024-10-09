

module load python-data

folder_path=$1
parent_dir="$(dirname $folder_path)"

for file in "$folder_path"/*; do
    if [ -f "$file" ]; then
        file_name=$(grep 'Training model: ' $file | sed 's/Training model: //')
        echo "Processing file: $file"
        python3  /scratch/project_2007095/attieh/All_distillation_methods/fairseq/scripts-joseph/plot_from_log/plot_data.py \
                $file \
                $parent_dir/plots/${file_name}.png

    fi
done