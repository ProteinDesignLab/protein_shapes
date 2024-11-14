import os
import argparse
import json
import os.path
import random
import sys

import numpy as np
import torch
from protein_shapes.proteinmpnn.data_utils import (
    alphabet,
    element_dict_rev,
    featurize,
    parse_PDB,
    restype_str_to_int,
)
from protein_shapes.proteinmpnn.model_utils import ProteinMPNN

def check_files(pdb, folder_for_outputs, model_type):
    
    # Get list of files in the directory
    prefix = ""
    if model_type == "ligand_mpnn":
        prefix = "ligand_"
    complete_pdbs = os.listdir(f"{prefix}{folder_for_outputs}")
    
    # Check if there is any file starting with the name
    find = pdb[len(folder_for_outputs) :-4] + "_"
    complete = any(complete_pdb.startswith(find) for complete_pdb in complete_pdbs)
    if complete:
        return False
    
    # If no file starts with the name and no file matches the error prefixes
    return True

def main(args) -> None:
    """
    Inference function
    """
    if args.seed:
        seed = args.seed
    else:
        seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    output_dir = args.out_folder[args.out_folder.find(' ')+1:]
    folder_for_outputs = args.out_folder[10:args.out_folder.find(' ')]
    base_folder = args.out_folder[:args.out_folder.find(' ')]
    if base_folder[-1] != "/":
        base_folder = base_folder + "/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)
    if not os.path.exists(base_folder + "seqs"):
        os.makedirs(base_folder + "seqs", exist_ok=True)
    if not os.path.exists(base_folder + "backbones"):
        os.makedirs(base_folder + "backbones", exist_ok=True)
    if not os.path.exists(base_folder + "packed"):
        os.makedirs(base_folder + "packed", exist_ok=True)
    if args.save_stats:
        if not os.path.exists(base_folder + "stats"):
            os.makedirs(base_folder + "stats", exist_ok=True)
    if args.model_type == "protein_mpnn":
        checkpoint_path = args.checkpoint_protein_mpnn
    elif args.model_type == "ligand_mpnn":
        checkpoint_path = args.checkpoint_ligand_mpnn
    elif args.model_type == "per_residue_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_per_residue_label_membrane_mpnn
    elif args.model_type == "global_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_global_label_membrane_mpnn
    elif args.model_type == "soluble_mpnn":
        checkpoint_path = args.checkpoint_soluble_mpnn
    else:
        print("Choose one of the available models")
        sys.exit()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if args.model_type == "ligand_mpnn":
        atom_context_num = checkpoint["atom_context_num"]
        ligand_mpnn_use_side_chain_context = args.ligand_mpnn_use_side_chain_context
        k_neighbors = checkpoint["num_edges"]
    else:
        atom_context_num = 1
        ligand_mpnn_use_side_chain_context = 0
        k_neighbors = checkpoint["num_edges"]

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=args.model_type,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if args.pdb_path_multi:
        with open(args.pdb_path_multi, "r") as fh:
            pdb_paths = list(json.load(fh))
    else:
        pdb_paths = [args.pdb_path]

    if args.fixed_residues_multi:
        with open(args.fixed_residues_multi, "r") as fh:
            fixed_residues_multi = json.load(fh)
    else:
        fixed_residues = [item for item in args.fixed_residues.split()]
        fixed_residues_multi = {}
        for pdb in pdb_paths:
            fixed_residues_multi[pdb] = fixed_residues

    if args.redesigned_residues_multi:
        with open(args.redesigned_residues_multi, "r") as fh:
            redesigned_residues_multi = json.load(fh)
    else:
        redesigned_residues = [item for item in args.redesigned_residues.split()]
        redesigned_residues_multi = {}
        for pdb in pdb_paths:
            redesigned_residues_multi[pdb] = redesigned_residues

    bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
    if args.bias_AA:
        tmp = [item.split(":") for item in args.bias_AA.split(",")]
        a1 = [b[0] for b in tmp]
        a2 = [float(b[1]) for b in tmp]
        for i, AA in enumerate(a1):
            bias_AA[restype_str_to_int[AA]] = a2[i]

    if args.bias_AA_per_residue_multi:
        with open(args.bias_AA_per_residue_multi, "r") as fh:
            bias_AA_per_residue_multi = json.load(
                fh
            )  # {"pdb_path" : {"A12": {"G": 1.1}}}
    else:
        if args.bias_AA_per_residue:
            with open(args.bias_AA_per_residue, "r") as fh:
                bias_AA_per_residue = json.load(fh)  # {"A12": {"G": 1.1}}
            bias_AA_per_residue_multi = {}
            for pdb in pdb_paths:
                bias_AA_per_residue_multi[pdb] = bias_AA_per_residue

    if args.omit_AA_per_residue_multi:
        with open(args.omit_AA_per_residue_multi, "r") as fh:
            omit_AA_per_residue_multi = json.load(
                fh
            )  # {"pdb_path" : {"A12": "PQR", "A13": "QS"}}
    else:
        if args.omit_AA_per_residue:
            with open(args.omit_AA_per_residue, "r") as fh:
                omit_AA_per_residue = json.load(fh)  # {"A12": "PG"}
            omit_AA_per_residue_multi = {}
            for pdb in pdb_paths:
                omit_AA_per_residue_multi[pdb] = omit_AA_per_residue
    omit_AA_list = args.omit_AA
    omit_AA = torch.tensor(
        np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32),
        device=device,
    )

    if len(args.parse_these_chains_only) != 0:
        parse_these_chains_only_list = args.parse_these_chains_only.split(",")
    else:
        parse_these_chains_only_list = []


    # loop over PDB paths
    prefix = ""
    if args.model_type == "ligand_mpnn":
        prefix = "ligand_"

    for pdb in pdb_paths:
        if args.verbose:
            print("Designing protein from this path:", pdb)
        if not os.path.isdir(f"{prefix}{folder_for_outputs}") or check_files(pdb,folder_for_outputs,args.model_type) or not args.pdb_path_multi:
            try:
                fixed_residues = fixed_residues_multi[pdb]
                redesigned_residues = redesigned_residues_multi[pdb]
                parse_all_atoms_flag = args.ligand_mpnn_use_side_chain_context or (
                    args.pack_side_chains and not args.repack_everything
                )

                protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
                    pdb,
                    device=device,
                    chains=parse_these_chains_only_list,
                    parse_all_atoms=parse_all_atoms_flag,
                    parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
                )

                # make chain_letter + residue_idx + insertion_code mapping to integers
                R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
                chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
                encoded_residues = []
                for i, R_idx_item in enumerate(R_idx_list):
                    tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
                    encoded_residues.append(tmp)
                encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
                encoded_residue_dict_rev = dict(
                    zip(list(range(len(encoded_residues))), encoded_residues)
                )

                bias_AA_per_residue = torch.zeros(
                    [len(encoded_residues), 21], device=device, dtype=torch.float32
                )
                if args.bias_AA_per_residue_multi or args.bias_AA_per_residue:
                    bias_dict = bias_AA_per_residue_multi[pdb]
                    for residue_name, v1 in bias_dict.items():
                        if residue_name in encoded_residues:
                            i1 = encoded_residue_dict[residue_name]
                            for amino_acid, v2 in v1.items():
                                if amino_acid in alphabet:
                                    j1 = restype_str_to_int[amino_acid]
                                    bias_AA_per_residue[i1, j1] = v2

                omit_AA_per_residue = torch.zeros(
                    [len(encoded_residues), 21], device=device, dtype=torch.float32
                )
                if args.omit_AA_per_residue_multi or args.omit_AA_per_residue:
                    omit_dict = omit_AA_per_residue_multi[pdb]
                    for residue_name, v1 in omit_dict.items():
                        if residue_name in encoded_residues:
                            i1 = encoded_residue_dict[residue_name]
                            for amino_acid in v1:
                                if amino_acid in alphabet:
                                    j1 = restype_str_to_int[amino_acid]
                                    omit_AA_per_residue[i1, j1] = 1.0

                fixed_positions = torch.tensor(
                    [int(item not in fixed_residues) for item in encoded_residues],
                    device=device,
                )
                redesigned_positions = torch.tensor(
                    [int(item not in redesigned_residues) for item in encoded_residues],
                    device=device,
                )

                # specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
                if args.transmembrane_buried:
                    buried_residues = [item for item in args.transmembrane_buried.split()]
                    buried_positions = torch.tensor(
                        [int(item in buried_residues) for item in encoded_residues],
                        device=device,
                    )
                else:
                    buried_positions = torch.zeros_like(fixed_positions)

                if args.transmembrane_interface:
                    interface_residues = [item for item in args.transmembrane_interface.split()]
                    interface_positions = torch.tensor(
                        [int(item in interface_residues) for item in encoded_residues],
                        device=device,
                    )
                else:
                    interface_positions = torch.zeros_like(fixed_positions)
                protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
                    1 - interface_positions
                ) + 1 * interface_positions * (1 - buried_positions)

                if args.model_type == "global_label_membrane_mpnn":
                    protein_dict["membrane_per_residue_labels"] = (
                        args.global_transmembrane_label + 0 * fixed_positions
                    )
                if len(args.chains_to_design) != 0:
                    chains_to_design_list = args.chains_to_design.split(",")
                else:
                    chains_to_design_list = protein_dict["chain_letters"]

                chain_mask = torch.tensor(
                    np.array(
                        [
                            item in chains_to_design_list
                            for item in protein_dict["chain_letters"]
                        ],
                        dtype=np.int32,
                    ),
                    device=device,
                )

                # create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
                if redesigned_residues:
                    protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
                elif fixed_residues:
                    protein_dict["chain_mask"] = chain_mask * fixed_positions
                else:
                    protein_dict["chain_mask"] = chain_mask

                if args.verbose:
                    PDB_residues_to_be_redesigned = [
                        encoded_residue_dict_rev[item]
                        for item in range(protein_dict["chain_mask"].shape[0])
                        if protein_dict["chain_mask"][item] == 1
                    ]
                    PDB_residues_to_be_fixed = [
                        encoded_residue_dict_rev[item]
                        for item in range(protein_dict["chain_mask"].shape[0])
                        if protein_dict["chain_mask"][item] == 0
                    ]
                    print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
                    print("These residues will be fixed: ", PDB_residues_to_be_fixed)

                # specify which residues are linked
                if args.symmetry_residues:
                    symmetry_residues_list_of_lists = [
                        x.split(",") for x in args.symmetry_residues.split("|")
                    ]
                    remapped_symmetry_residues = []
                    for t_list in symmetry_residues_list_of_lists:
                        tmp_list = []
                        for t in t_list:
                            tmp_list.append(encoded_residue_dict[t])
                        remapped_symmetry_residues.append(tmp_list)
                else:
                    remapped_symmetry_residues = [[]]

                # specify linking weights
                if args.symmetry_weights:
                    symmetry_weights = [
                        [float(item) for item in x.split(",")]
                        for x in args.symmetry_weights.split("|")
                    ]
                else:
                    symmetry_weights = [[]]

                if args.homo_oligomer:
                    if args.verbose:
                        print("Designing HOMO-OLIGOMER")
                    chain_letters_set = list(set(chain_letters_list))
                    reference_chain = chain_letters_set[0]
                    lc = len(reference_chain)
                    residue_indices = [
                        item[lc:] for item in encoded_residues if item[:lc] == reference_chain
                    ]
                    remapped_symmetry_residues = []
                    symmetry_weights = []
                    for res in residue_indices:
                        tmp_list = []
                        tmp_w_list = []
                        for chain in chain_letters_set:
                            name = chain + res
                            tmp_list.append(encoded_residue_dict[name])
                            tmp_w_list.append(1 / len(chain_letters_set))
                        remapped_symmetry_residues.append(tmp_list)
                        symmetry_weights.append(tmp_w_list)

                # set other atom bfactors to 0.0
                if other_atoms:
                    other_bfactors = other_atoms.getBetas()
                    other_atoms.setBetas(other_bfactors * 0.0)

                # adjust input PDB name by dropping .pdb if it does exist
                name = pdb[pdb.rfind("/") + 1 :]
                if name.endswith(".pdb"):  # Check if the filename ends with ".pdb"
                    name = name.replace(".pdb", "")  # Remove the ".pdb" extension

                with torch.no_grad():
                    # run featurize to remap R_idx and add batch dimension
                    if args.verbose:
                        if "Y" in list(protein_dict):
                            atom_coords = protein_dict["Y"].cpu().numpy()
                            atom_types = list(protein_dict["Y_t"].cpu().numpy())
                            atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                            number_of_atoms_parsed = np.sum(atom_mask)
                        else:
                            print("No ligand atoms parsed")
                            number_of_atoms_parsed = 0
                            atom_types = ""
                            atom_coords = []
                        if number_of_atoms_parsed == 0:
                            print("No ligand atoms parsed")
                        elif args.model_type == "ligand_mpnn":
                            print(
                                f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                            )
                            for i, atom_type in enumerate(atom_types):
                                print(
                                    f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                                )
                    feature_dict = featurize(
                        protein_dict,
                        cutoff_for_score=args.ligand_mpnn_cutoff_for_score,
                        use_atom_context=args.ligand_mpnn_use_atom_context,
                        number_of_ligand_atoms=atom_context_num,
                        model_type=args.model_type,
                    )
                    feature_dict["batch_size"] = args.batch_size
                    B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
                    # add additional keys to the feature dictionary
                    feature_dict["temperature"] = args.temperature
                    feature_dict["bias"] = (
                        (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                        + bias_AA_per_residue[None]
                        - 1e8 * omit_AA_per_residue[None]
                    )
                    feature_dict["symmetry_residues"] = remapped_symmetry_residues
                    feature_dict["symmetry_weights"] = symmetry_weights

                    for _ in range(args.number_of_batches):
                        feature_dict["randn"] = torch.randn(
                            [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                            device=device,
                        )
                        output_dict = model.sample(feature_dict,pdb=name,output=folder_for_outputs,output_dir=output_dir)
                        
            except Exception as e:
                print(f"Error occurred while generating LigandMPNN embedding for {pdb}: {e}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument(
        "--model_type",
        type=str,
        default="protein_mpnn",
        help="Choose your model: protein_mpnn, ligand_mpnn, per_residue_label_membrane_mpnn, global_label_membrane_mpnn, soluble_mpnn",
    )
    # protein_mpnn - original ProteinMPNN trained on the whole PDB exluding non-protein atoms
    # ligand_mpnn - atomic context aware model trained with small molecules, nucleotides, metals etc on the whole PDB
    # per_residue_label_membrane_mpnn - ProteinMPNN model trained with addition label per residue specifying if that residue is buried or exposed
    # global_label_membrane_mpnn - ProteinMPNN model trained with global label per PDB id to specify if protein is transmembrane
    # soluble_mpnn - ProteinMPNN trained only on soluble PDB ids
    argparser.add_argument(
        "--checkpoint_protein_mpnn",
        type=str,
        default="./model_params/proteinmpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_ligand_mpnn",
        type=str,
        default="./model_params/ligandmpnn_v_32_010_25.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_per_residue_label_membrane_mpnn",
        type=str,
        default="./model_params/per_residue_label_membrane_mpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_global_label_membrane_mpnn",
        type=str,
        default="./model_params/global_label_membrane_mpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_soluble_mpnn",
        type=str,
        default="./model_params/solublempnn_v_48_020.pt",
        help="Path to model weights.",
    )

    argparser.add_argument(
        "--fasta_seq_separation",
        type=str,
        default=":",
        help="Symbol to use between sequences from different chains",
    )
    argparser.add_argument("--verbose", type=int, default=1, help="Print stuff")

    argparser.add_argument(
        "--pdb_path", type=str, default="", help="Path to the input PDB."
    )
    argparser.add_argument(
        "--pdb_path_multi",
        type=str,
        default="",
        help="Path to json listing PDB paths. {'/path/to/pdb': ''} - only keys will be used.",
    )

    argparser.add_argument(
        "--fixed_residues",
        type=str,
        default="",
        help="Provide fixed residues, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--fixed_residues_multi",
        type=str,
        default="",
        help="Path to json mapping of fixed residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    argparser.add_argument(
        "--redesigned_residues",
        type=str,
        default="",
        help="Provide to be redesigned residues, everything else will be fixed, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--redesigned_residues_multi",
        type=str,
        default="",
        help="Path to json mapping of redesigned residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    argparser.add_argument(
        "--bias_AA",
        type=str,
        default="",
        help="Bias generation of amino acids, e.g. 'A:-1.024,P:2.34,C:-12.34'",
    )
    argparser.add_argument(
        "--bias_AA_per_residue",
        type=str,
        default="",
        help="Path to json mapping of bias {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}",
    )
    argparser.add_argument(
        "--bias_AA_per_residue_multi",
        type=str,
        default="",
        help="Path to json mapping of bias {'pdb_path': {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}}",
    )

    argparser.add_argument(
        "--omit_AA",
        type=str,
        default="",
        help="Bias generation of amino acids, e.g. 'ACG'",
    )
    argparser.add_argument(
        "--omit_AA_per_residue",
        type=str,
        default="",
        help="Path to json mapping of bias {'A12': 'APQ', 'A13': 'QST'}",
    )
    argparser.add_argument(
        "--omit_AA_per_residue_multi",
        type=str,
        default="",
        help="Path to json mapping of bias {'pdb_path': {'A12': 'QSPC', 'A13': 'AGE'}}",
    )

    argparser.add_argument(
        "--symmetry_residues",
        type=str,
        default="",
        help="Add list of lists for which residues need to be symmetric, e.g. 'A12,A13,A14|C2,C3|A5,B6'",
    )
    argparser.add_argument(
        "--symmetry_weights",
        type=str,
        default="",
        help="Add weights that match symmetry_residues, e.g. '1.01,1.0,1.0|-1.0,2.0|2.0,2.3'",
    )
    argparser.add_argument(
        "--homo_oligomer",
        type=int,
        default=0,
        help="Setting this to 1 will automatically set --symmetry_residues and --symmetry_weights to do homooligomer design with equal weighting.",
    )

    argparser.add_argument(
        "--out_folder",
        type=str,
        help="Path to a folder to output sequences, e.g. /home/out/",
    )
    argparser.add_argument(
        "--file_ending", type=str, default="", help="adding_string_to_the_end"
    )
    argparser.add_argument(
        "--zero_indexed",
        type=str,
        default=0,
        help="1 - to start output PDB numbering with 0",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set seed for torch, numpy, and python random.",
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of sequence to generate per one pass.",
    )
    argparser.add_argument(
        "--number_of_batches",
        type=int,
        default=1,
        help="Number of times to design sequence using a chosen batch size.",
    )
    argparser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature to sample sequences.",
    )
    argparser.add_argument(
        "--save_stats", type=int, default=0, help="Save output statistics"
    )

    argparser.add_argument(
        "--ligand_mpnn_use_atom_context",
        type=int,
        default=1,
        help="1 - use atom context, 0 - do not use atom context.",
    )
    argparser.add_argument(
        "--ligand_mpnn_cutoff_for_score",
        type=float,
        default=8.0,
        help="Cutoff in angstroms between protein and context atoms to select residues for reporting score.",
    )
    argparser.add_argument(
        "--ligand_mpnn_use_side_chain_context",
        type=int,
        default=0,
        help="Flag to use side chain atoms as ligand context for the fixed residues",
    )
    argparser.add_argument(
        "--chains_to_design",
        type=str,
        default="",
        help="Specify which chains to redesign, all others will be kept fixed, 'A,B,C,F'",
    )

    argparser.add_argument(
        "--parse_these_chains_only",
        type=str,
        default="",
        help="Provide chains letters for parsing backbones, 'A,B,C,F'",
    )

    argparser.add_argument(
        "--transmembrane_buried",
        type=str,
        default="",
        help="Provide buried residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--transmembrane_interface",
        type=str,
        default="",
        help="Provide interface residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )

    argparser.add_argument(
        "--global_transmembrane_label",
        type=int,
        default=0,
        help="Provide global label for global_label_membrane_mpnn model. 1 - transmembrane, 0 - soluble",
    )

    argparser.add_argument(
        "--parse_atoms_with_zero_occupancy",
        type=int,
        default=0,
        help="To parse atoms with zero occupancy in the PDB input files. 0 - do not parse, 1 - parse atoms with zero occupancy",
    )

    argparser.add_argument(
        "--pack_side_chains",
        type=int,
        default=0,
        help="1 - to run side chain packer, 0 - do not run it",
    )

    argparser.add_argument(
        "--checkpoint_path_sc",
        type=str,
        default="./model_params/ligandmpnn_sc_v_32_002_16.pt",
        help="Path to model weights.",
    )

    argparser.add_argument(
        "--number_of_packs_per_design",
        type=int,
        default=4,
        help="Number of independent side chain packing samples to return per design",
    )

    argparser.add_argument(
        "--sc_num_denoising_steps",
        type=int,
        default=3,
        help="Number of denoising/recycling steps to make for side chain packing",
    )

    argparser.add_argument(
        "--sc_num_samples",
        type=int,
        default=16,
        help="Number of samples to draw from a mixture distribution and then take a sample with the highest likelihood.",
    )

    argparser.add_argument(
        "--repack_everything",
        type=int,
        default=0,
        help="1 - repacks side chains of all residues including the fixed ones; 0 - keeps the side chains fixed for fixed residues",
    )

    argparser.add_argument(
        "--force_hetatm",
        type=int,
        default=0,
        help="To force ligand atoms to be written as HETATM to PDB file after packing.",
    )

    argparser.add_argument(
        "--packed_suffix",
        type=str,
        default="_packed",
        help="Suffix for packed PDB paths",
    )

    argparser.add_argument(
        "--pack_with_ligand_context",
        type=int,
        default=1,
        help="1-pack side chains using ligand context, 0 - do not use it.",
    )

    args = argparser.parse_args()
    main(args)

