# download and prep training data
## useful function from https://hackersandslackers.com/extract-data-from-complex-json-python/
import re
import json
import pandas as pd
import numpy as np
import os   
import ast
import collections


def json_extract(obj, key):
    """Recursively fetch values from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    return values

# function to tokenize PFAMs
def tokenize_pfam(pfam, with_begin_and_end=True):
    """
    Tokenizes a PFAM string
    :param pfam: a string of space delimited PFAM domains.
    :param with_begin_and_end: Appends a begin token and prepends an end token.
    :return : A string with the tokenized version.
    """
    tokens = pfam.split(' ')
    #if with_begin_and_end:
    #    tokens = ["[START]"] + tokens + ["[END]"]
    tokens = " ".join(tokens)
    return tokens

# function to tokenize SMILES
# see https://github.com/undeadpixel/reinvent-randomized/blob/master/models/vocabulary.py

REGEXPS = {
    "brackets": re.compile(r"(\[[^\]]*\])"),
    "2_ring_nums": re.compile(r"(%\d{2})"),
    "brcl": re.compile(r"(Br|Cl)")
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
    #if with_begin_and_end:
    #    tokens = ["[START]"] + tokens + ["[END]"]
    tokens = "".join(tokens)
    return tokens

# function to load data   
def data_loader(filename):
    # load joined data
    joined_df_path = filename
    joined_df = pd.read_csv(joined_df_path, index_col="accession", sep="\t")

    # get inp1
    tokenized_pfams_list = joined_df['pfams'].map(lambda pfam: tokenize_pfam(pfam) if pd.notnull(pfam) else np.nan).tolist()

    # get inp2
    tokenized_draft_smiles_list = joined_df['antismash_struct'].map(lambda smiles: tokenize_smiles(smiles) if pd.notnull(smiles) else np.nan).tolist()

    # get targ
    tokenized_target_smiles_list = joined_df['chem_struct'].map(lambda smiles: tokenize_smiles(smiles) if pd.notnull(smiles) else np.nan).tolist()
    return tokenized_pfams_list, tokenized_draft_smiles_list, tokenized_target_smiles_list