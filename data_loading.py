import pandas as pd
from Bio.PDB import PDBList, PDBParser
from Bio.PDB.Polypeptide import is_aa

# Amino acid 3-letter to 1-letter code dictionary
aa_3to1 = { "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G",
            "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
            "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V" }

def get_pdb_sequence_and_b_factors(pdb_id: str):
    """Fetches sequence and B-factors for a PDB ID.
    
    Args:
        pdb_id (str): The PDB ID for the protein structure.

    Returns:
        tuple: (sequences, b_factors), lists of sequences and B-factors dictionaries for each chain.
    """
    pdbl = PDBList()
    try:
        pdb_file = pdbl.retrieve_pdb_file(pdb_id, file_format="pdb")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_file)

        sequences, b_factors = [], []
        for model in structure:
            for chain in model:
                chain_seq, avg_b_factors, ca_b_factors = "", [], []
                for residue in chain.get_residues():
                    if is_aa(residue, standard=True):
                        resname = residue.resname.strip()
                        chain_seq += aa_3to1.get(resname, "")
                        all_b_factors = [atom.get_bfactor() for atom in residue]
                        avg_b_factors.append(sum(all_b_factors) / len(all_b_factors))
                        ca_b_factors.append(next((atom.get_bfactor() for atom in residue if atom.get_name() == "CA"), None))

                sequences.append(chain_seq)
                b_factors.append({"avg_b_factor": avg_b_factors, "ca_b_factor": ca_b_factors})

        return sequences, b_factors
    except Exception as e:
        print(f"Error fetching PDB {pdb_id}: {e}")
        return [], []

def create_dataframe(pdb_ids):
    """Creates a DataFrame with sequences and B-factors for multiple PDB IDs."""
    data = []
    for pdb_id in pdb_ids:
        sequences, b_factors = get_pdb_sequence_and_b_factors(pdb_id)
        for seq, b_factor in zip(sequences, b_factors):
            data.append({"pdb_id": pdb_id, "sequence": seq, "avg_b_factor": b_factor["avg_b_factor"], "ca_b_factor": b_factor["ca_b_factor"]})
    return pd.DataFrame(data)
