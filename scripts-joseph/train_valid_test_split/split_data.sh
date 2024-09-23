#!/bin/bash

# Define file and index files
file=$1
dir=$2 

train_indexes="$2/train_indexes.txt"
valid_indexes="$2/valid_indexes.txt"
test_indexes="$2/test_indexes.txt"


file_base="$(basename $1 )"
# Split files using indexes


last_prefix="$(echo $file_base | cut -d '.' -f2,3)"

awk 'NR==FNR{a[$1]; next} FNR in a' "$train_indexes" "$file" > $2/train.$last_prefix
awk 'NR==FNR{a[$1]; next} FNR in a' "$valid_indexes" "$file" > $2/valid.$last_prefix
awk 'NR==FNR{a[$1]; next} FNR in a' "$test_indexes" "$file" > $2/test.$last_prefix

echo "File split into: train.txt, valid.txt, test.txt"
