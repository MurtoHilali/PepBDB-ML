'''
    A series of hellper functions designed to create an enriched version
    of the PepBDB database, ideal for machine learning and CNNs.
'''
import numpy as np
import os
from Bio.PDB import PDBParser, HSExposure, DSSP
from Bio.SeqUtils import seq1
import subprocess
import pandas as pd
from ast import literal_eval
import tempfile
from paths import *
from typing import List

def extract_sequence(pdb_filename: str) -> str:
    '''
    Helper function that extracts the sequence from a PDB file.
    '''
    parser = PDBParser()
    structure = parser.get_structure('X', pdb_filename)
    
    for model in structure:
        for chain in model:
            residues = chain.get_residues()
            sequence = ''
            for residue in residues:
                if residue.get_resname() == 'HOH': # ignoring water
                    continue
                sequence += seq1(residue.get_resname()) # seq1 converts 3-letter code to 1-letter code
            return sequence

def label_residues(peptide_path: str, protein_path: str) -> List:
    # Creating a temporary file with the peptide and protein
    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_file:
        output_path = temp_file.name
        subprocess.run(['cat', peptide_path, protein_path], stdout=temp_file)

    # Ensuring the temp file is properly closed before using it in subprocess
    temp_file.close()
    
    # Obtaining binding residues using PRODIGY
    subprocess.run(['prodigy', '-q', '--contact_list', output_path])

    # The .ic file will have the same root name as the input file
    ic_file_path = output_path.replace('.pdb', '.ic')

    # Check if the .ic file exists before reading
    if not os.path.exists(ic_file_path):
        raise FileNotFoundError(f"{ic_file_path} not found after running PRODIGY.")

    contacts = pd.read_csv(ic_file_path, sep='\s+', header=None)
    contacts.columns = ['peptide_residue', 'peptide_index', 'peptide_chain', 'protein_residue', 'protein_index', 'protein_chain']
    
    peptide_binding_residues = sorted(list(contacts['peptide_index'].unique()))
    protein_binding_residues = sorted(list(contacts['protein_index'].unique()))
    
    os.remove(output_path)
    
    return peptide_binding_residues, protein_binding_residues

def hse_and_dssp(pdb_file: str):
    '''
    Get the HSE and DSSP codes of a PDB's residues.
    '''
    # Initialize PDBParser
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure('structure', pdb_file)
    
    hse = HSExposure.HSExposureCA(structure)
    dssp = DSSP(structure[0], pdb_file)
    
    # Half-sphere exposure values
    hse_up, hse_down, pseudo_angle = zip(*[(res[1][0], res[1][1], res[1][2]) for res in hse.property_list])
   
    # Extract DSSP values into separate lists
    ss, asa, phi, psi = zip(*[(res[2], res[3], res[4], res[5]) for res in dssp.property_list])
    
    print(f'\rData acquired for {pdb_file}')
    return hse_up, hse_down, pseudo_angle, ss, asa, phi, psi

def safe_hse_and_dssp(pdb_file):
    '''
    Helper function to handle errors and apply `hse_and_dssp`.
    '''
    
    # Empty list to store error-producing filenames
    error_files = []
    try:
        return hse_and_dssp(pdb_file)
    except Exception as e:
        print(f'Error processing file: {pdb_file} - {e}')
        error_files.append(pdb_file)
        return [None] * 7  # Return a list of None values to match the expected output structure

# Possible DSSP values
dssp_codes = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']

# One-hot encoding function
def one_hot_encode_array(ss_array):
    ss_array = literal_eval(str(ss_array))
    length = len(ss_array)
    encoding = {code: [0] * length for code in dssp_codes}
    for i, code in enumerate(ss_array):
        encoding[code][i] = 1
    return encoding

def one_hot_encode_row(row):
    pep_encoded = one_hot_encode_array(row['Peptide SS'])
    prot_encoded = one_hot_encode_array(row['Protein SS'])
    new_data = {}
    for code in dssp_codes:
        new_data[f'Peptide SS {code}'] = pep_encoded[code]
        new_data[f'Protein SS {code}'] = prot_encoded[code]
    return pd.Series(new_data)

def extend_hse(hse: str) -> str:
    """
    Extends a HSE to the full length of the peptide.
    """
    hse = list(literal_eval(str(hse)))
    hse = [hse[0]] + hse + [hse[-1]]
    
    return hse

def get_pssm_profile(sequence: str) -> pd.DataFrame:
    '''
    Uses blast+ psiblast to generate PSSM profile from a 
    temporary fasta file.
    '''
    with tempfile.NamedTemporaryFile(suffix='.fa', delete=False) as fasta_file:
        fasta_file.write(f'>tmp\n{sequence}'.encode('utf-8'))
        fasta_path = fasta_file.name
    
    pssm_fd, pssm_path = tempfile.mkstemp(suffix='.pssm')
    os.close(pssm_fd)

    try:
        subprocess.run(f'psiblast -query {fasta_path} -db {swissprot} -num_iterations 3 -evalue 0.001 -out_ascii_pssm {pssm_path} 2>/dev/null', shell=True, check=True)
        
        with open(pssm_path, 'r') as file:
            lines = file.readlines()
        
        # Skip the header lines and parse the matrix
        pssm_data = []
        for line in lines[3:len(sequence) + 3]:
            parts = line.strip().split()
            scores = parts[1:22]  # First 20 columns are scores for each amino acid
            pssm_data.append(scores)
        
        columns = ['AA'] + list('ARNDCQEGHILKMFPSTWYV')
        df_pssm = pd.DataFrame(pssm_data, columns=columns)
        
    finally:
        os.remove(fasta_path)
        os.remove(pssm_path)
    
    return df_pssm.T

def make_tabular_dataset(row: pd.Series) -> pd.DataFrame: 
    '''
    `make_tabular_dataset` is designed to be applied to a row of the input DataFrame
    to create a feature array.
    
    '''
    feature_dict = row.to_dict()
    
    # get sequence and PSSM
    sequence = feature_dict[f'Protein Sequence']
    pssm = feature_dict[f'Protein PSSM']
    
    # remove the first row of the PSSM, which is the sequence
    pssm = pssm.reset_index(drop=True)
    pssm = pssm.iloc[1:]
    
    # get the binding indices list
    binding_indices_dummy = [0] * len(sequence)
    binding_indices = feature_dict[f'Protein Binding Indices']
    for i in range(len(sequence)):
        if i+1 in literal_eval(binding_indices):
            binding_indices_dummy[i] = 1
    
    # add the sequence as the first element of the array
    arr = []
    arr.append([char for char in sequence]) 
    
    # remove these columns from the dictionary; all are either unneeded or have been processed
    remove = ['Unnamed: 0', 'PDB ID',
              'Number of Atoms in Protein', 'Protein Chain ID', 'Protein Path',
              'Number of Atom Contacts', 'Resolution',
              'Molecular Type', 'Protein Path', 'Protein Sequence',
              'Protein Binding Indices', 'Protein SS', 'Protein PSSM'] 
    
    # many of the features are stored as strings, so we need to convert them to lists
    use_feature_dict = {}
    for k, v in feature_dict.items():
        if k not in remove:
            try:
                use_feature_dict[k] = literal_eval(str(v))
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing key '{k}' with value '{v}': {e}")

    for key, value in use_feature_dict.items():
        value_array = [residue_value for residue_value in value]
        arr.append(value_array) # add the feature to the array
    
    # stack the arrays on top of each other
    try:
        arr = pd.DataFrame(np.vstack((arr, pssm, binding_indices_dummy)))
    except ValueError as e:
        print("An error occurred:", e)
    
    # transpose the array and add the column names
    arr = pd.DataFrame(arr).T
    columns = ['AA'] + list(use_feature_dict.keys()) + list('ARNDCQEGHILKMFPSTWYV') + ['Binding Indices']
    
    if len(arr.columns) == len(columns):
        arr.columns = columns
    else:
        print("Column length mismatch")
        print("Arr shape:", arr.shape)
        print("Columns length:", len(columns))

    return pd.DataFrame(arr)