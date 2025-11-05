#!/bin/bash
set -e  # Stop on first error

BASE_DIR="/home/saidkoussi/Downloads"
OUTPUT_DIR="${BASE_DIR}/unzipped_files"

mkdir -p "$OUTPUT_DIR"

echo "=== Step 1: Searching for .zip files in ${BASE_DIR} ==="
zip_files=($(find "$BASE_DIR" -maxdepth 1 -type f -name "*.zip"))

if [ ${#zip_files[@]} -eq 0 ]; then
    echo " No zip files found in ${BASE_DIR}."
    exit 1
fi

echo "=== Step 2: Unzipping files ==="
for zip_file in "${zip_files[@]}"; do
    folder_name="${OUTPUT_DIR}/$(basename "${zip_file%.zip}")"
    mkdir -p "$folder_name"
    echo "Unzipping: $zip_file → $folder_name"
    unzip -o "$zip_file" -d "$folder_name" >/dev/null
done

echo "=== Step 3: Collecting all .npz and .csv files into ${OUTPUT_DIR}/collected ==="
COLLECTED_DIR="${OUTPUT_DIR}/collected"
mkdir -p "$COLLECTED_DIR"

find "$OUTPUT_DIR" -type f \( -name "*.npz" -o -name "*.csv" \) -exec cp {} "$COLLECTED_DIR" \;

echo " All .zip files unzipped and .npz/.csv files collected in:"
echo " $COLLECTED_DIR"
