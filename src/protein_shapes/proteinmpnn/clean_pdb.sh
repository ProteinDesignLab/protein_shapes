#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <directory_path>"
  exit 1
fi

# Directory containing the PDB files
pdb_directory="$1"

# Check if the directory exists
if [ ! -d "$pdb_directory" ]; then
  echo "Directory does not exist: $pdb_directory"
  exit 1
fi

# Iterate over all .pdb files in the specified directory
for pdb_file in "$pdb_directory"/*.pdb; do
  # Use pdb_selaltloc to process each file and temporarily save the output
  pdb_selaltloc "$pdb_file" > "${pdb_file%.pdb}_temp.pdb"
  # Overwrite the original file with the modified one
  mv "${pdb_file%.pdb}_temp.pdb" "$pdb_file"
done

echo "Processing complete in directory $pdb_directory."
