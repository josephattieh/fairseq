#!/bin/bash

# Define file and total line count
file=$1
dir=$2
total_lines=$(wc -l < "$file")

# Set percentages for train, validation, and test splits
train_pct=70
valid_pct=15
test_pct=15

# Calculate the number of lines for each split
train_count=$((total_lines * train_pct / 100))
valid_count=$((total_lines * valid_pct / 100))
test_count=$((total_lines - train_count - valid_count))

# Shuffle the file line indexes and split them
shuf -i 1-$total_lines > shuffled_indexes.txt

# Generate train, valid, and test index files
head -n "$train_count" shuffled_indexes.txt > $2/train_indexes.txt
tail -n +"$((train_count + 1))" shuffled_indexes.txt | head -n "$valid_count" > $2/valid_indexes.txt
tail -n "$test_count" shuffled_indexes.txt > $2/test_indexes.txt

# Clean up temporary file
rm shuffled_indexes.txt

echo "Indexes generated: train_indexes.txt, valid_indexes.txt, test_indexes.txt"
