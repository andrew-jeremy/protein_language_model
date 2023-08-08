"""Utilities for working with SMILES strings."""

import re

from rdkit import Chem

REGEXPS = {
    "brackets": re.compile(r"(\[[^\]]*\])"),
    "2_ring_nums": re.compile(r"(%\d{2})"),
    "brcl": re.compile(r"(Br|Cl)"),
}
REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]


def tokenize_smiles(smiles, with_begin_and_end=True):
    """
    Tokenizes a SMILES string.
    :param smiles: A SMILES string.
    :param with_begin_and_end: Appends a begin token and prepends an end token.
    :return : A string with the tokenized version.
    """

    def split_by(smiles, regexps):
        if not regexps:
            return list(smiles)
        regexp = REGEXPS[regexps[0]]
        splitted = regexp.split(smiles)
        tokens = []
        for i, split in enumerate(splitted):
            if i % 2 == 0:
                tokens += split_by(split, regexps[1:])
            else:
                tokens.append(split)
        return tokens

    tokens = split_by(smiles, REGEXP_ORDER)
    if with_begin_and_end:
        tokens = ["[START]"] + tokens + ["[END]"]
    tokens = " ".join(tokens)
    return tokens


def is_valid_smiles(smiles):
    """Determine if a SMILES string is valid.

    :param smiles: SMILES string
    :type smiles: str
    :return: True if the SMILES string is valid, False otherwise
    :rtype: bool
    """
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return False
        return True
    except Exception:
        return False


def get_unique_molecules(smiles_list):
    """Returns a list of unique molecules from a list of SMILES strings.

    :param smiles_list: a list of SMILES strings
    :type smiles_list: list
    :return unique_molecules: a list of unique molecules
    :rtype: list
    """
    unique_molecules = []
    unique_smiles = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            if canonical_smiles not in unique_smiles:
                unique_molecules.append(mol)
                unique_smiles.add(canonical_smiles)
    return unique_molecules


def canonicalize_smiles(smiles):
    """Canonicalize a SMILES string.

    :param smiles: SMILES string
    :type smiles: str
    :return: Canonicalized SMILES string
    :rtype: str
    """
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)


def randomize_smiles(smiles):
    """Generate a random, noncanonical representation of a given smiles

    :param smiles: smiles string
    :returns: randomized smiles string
    """
    mol = Chem.MolFromSmiles(smiles)
    rand_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
    return rand_smiles

