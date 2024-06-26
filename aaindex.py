def feature_vector(seq: str, feature_type: dict):
    '''
    Returns a list of feature values corresponding to each
    residue in sequence, according to AAindex1.
    
    Feature types include:
    - Hydrophobicity
    - Steric parameter
    - Residue volume
    - Polarizability
    - Average relative probability of helix
    - Average relative probability of beta
    - Isoelectric point
    '''
    return [feature_type[aa] for aa in seq]

hydrophobicity = {
    'A': 0.61,
    'L': 1.53,
    'R': 0.60,
    'K': 1.15,
    'N': 0.06,
    'M': 1.18,
    'D': 0.46,
    'F': 2.02,
    'C': 1.07,
    'P': 1.95,
    'Q': 0.00,
    'S': 0.05,
    'E': 0.47,
    'T': 0.05,
    'G': 0.07,
    'W': 2.65,
    'H': 0.61,
    'Y': 1.88,
    'I': 2.22,
    'V': 1.32
}

steric_parameter = {
    'A': 0.52,
    'L': 0.98,
    'R': 0.68,
    'K': 0.68,
    'N': 0.76,
    'M': 0.78,
    'D': 0.76,
    'F': 0.70,
    'C': 0.62,
    'P': 0.36,
    'Q': 0.68,
    'S': 0.53,
    'E': 0.68,
    'T': 0.50,
    'G': 0.00,
    'W': 0.70,
    'H': 0.70,
    'Y': 0.70,
    'I': 1.02,
    'V': 0.76
}

residue_volume = {
    'A': 52.6,
    'L': 102.0,
    'R': 109.1,
    'K': 105.1,
    'N': 75.7,
    'M': 97.7,
    'D': 68.4,
    'F': 113.9,
    'C': 68.3,
    'P': 73.6,
    'Q': 89.7,
    'S': 54.9,
    'E': 84.7,
    'T': 71.2,
    'G': 36.3,
    'W': 135.4,
    'H': 91.9,
    'Y': 116.2,
    'I': 102.0,
    'V': 85.1
}

polarizability = {
    'A': 0.046,
    'L': 0.186,
    'R': 0.291,
    'K': 0.219,
    'N': 0.134,
    'M': 0.221,
    'D': 0.105,
    'F': 0.290,
    'C': 0.128,
    'P': 0.131,
    'Q': 0.180,
    'S': 0.062,
    'E': 0.151,
    'T': 0.108,
    'G': 0.000,
    'W': 0.409,
    'H': 0.230,
    'Y': 0.298,
    'I': 0.186,
    'V': 0.140
}

average_relative_probability_of_helix = {
    'A': 1.36,
    'L': 1.21,
    'R': 1.00,
    'K': 1.22,
    'N': 0.89,
    'M': 1.45,
    'D': 1.04,
    'F': 1.05,
    'C': 0.82,
    'P': 0.52,
    'Q': 1.14,
    'S': 0.74,
    'E': 1.48,
    'T': 0.81,
    'G': 0.63,
    'W': 0.97,
    'H': 1.11,
    'Y': 0.79,
    'I': 1.08,
    'V': 0.94
}

average_relative_probability_of_beta_sheet = {
    'A': 0.81,
    'L': 1.24,
    'R': 0.85,
    'K': 0.77,
    'N': 0.62,
    'M': 1.05,
    'D': 0.71,
    'F': 1.20,
    'C': 1.17,
    'P': 0.61,
    'Q': 0.98,
    'S': 0.92,
    'E': 0.53,
    'T': 1.18,
    'G': 0.88,
    'W': 1.18,
    'H': 0.92,
    'Y': 1.23,
    'I': 1.48,
    'V': 1.66
}

isoelectric_point = {
    'A': 6.00,
    'L': 5.98,
    'R': 10.76,
    'K': 9.74,
    'N': 5.41,
    'M': 5.74,
    'D': 2.77,
    'F': 5.48,
    'C': 5.05,
    'P': 6.30,
    'Q': 5.65,
    'S': 5.68,
    'E': 3.22,
    'T': 5.66,
    'G': 5.97,
    'W': 5.89,
    'H': 7.59,
    'Y': 5.66,
    'I': 6.02,
    'V': 5.96
}