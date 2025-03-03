import argparse
from pathlib import Path
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Compute LigandMPNN or ProteinMPNN embeddings."
    )
    parser.add_argument("directory", help="Directory to process")
    parser.add_argument("output_dir", help="Directory to output embeddings")
    args = parser.parse_args()

    # Preprocessing commands
    cur_dir = Path(__file__).parent

    clean_script = cur_dir / "clean_pdb.sh"
    generate_json_script = cur_dir / "generate_pdb_json.py"
    run_script = cur_dir / "run.py"

    clean_command = [str(clean_script), args.directory]
    generate_json_command = [sys.executable, str(generate_json_script), args.directory]
    # print(f"gen command: {generate_json_command}")

    # subprocess.run(clean_command, check=True)
    subprocess.run(generate_json_command, check=True)

    directory = (
        str(args.directory)[1:]
        if str(args.directory).startswith("/")
        else str(args.directory)
    ).replace("/", "-")
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    json_path = cur_dir / f"{directory}_output.json"

    checkpoint_path = cur_dir / "model_params/proteinmpnn_v_48_020.pt"
    main_command = f"python {run_script} --seed 111 --pdb_path_multi {json_path} --out_folder '{output_dir}' --checkpoint_protein_mpnn {checkpoint_path} --parse_atoms_with_zero_occupancy 1 --verbose 0"

    os.system(main_command)

    os.remove(json_path)


if __name__ == "__main__":
    main()
