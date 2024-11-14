import os
import sys
from pathlib import Path


def generate_json(directory):

    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    pdb_files = {}

    for filename in os.listdir(directory):
        file_path = os.path.join("./", directory, filename)
        pdb_files[file_path] = ""

    directory = (
        str(directory)[1:] if str(directory).startswith("/") else str(directory)
    ).replace("/", "-")
    json_output_path = os.path.join(Path(__file__).parent, directory + "_output.json")

    with open(json_output_path, "w") as json_file:
        json_file.write("{\n")
        entries = [f'"{path}": ""' for path in pdb_files]
        json_file.write(",\n".join(entries))
        json_file.write("\n}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    generate_json(sys.argv[1])
