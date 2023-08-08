# Canonize smiles and generate a vocabulary set for all the smiles in the dataset
# Run this module once to create new smiles tokens
# There are two approaches used: 1. treats all characters as single tokens, 2.treats all atoms as tokens
# run this only once to generate a vocabulary. Vocabs need to remain the same for train & inference
# Andrew Kiruluta

import pandas as pd
import pickle
from rdkit import Chem
from pysmilesutils.tokenize import *
from pysmilesutils.analyze import analyze_smiles_tokens

def canonize(x):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(x))
    except:
        return x

# 1. treat all characters as individual tokens and create vocabulary "smiles_con_vocab.pkl"
df1 = pd.read_csv('data/df_targ.csv')
df = df1[df1['target'].notna()]
#df2 = pd.read_csv('data/df_inp2.csv')
#df = pd.concat([df1, df2], ignore_index=True, sort=False)

df['canon_seq'] = df['target'].apply(lambda x: canonize(x))
smiles = df.canon_seq.tolist()
y = []
for i in smiles:
    try:
        y.append(list(i))
    except:
        continue
vocab_target = [item for sublist in y for item in sublist]
smiles_vocab_target = list(set(vocab_target))
smiles_vocab_target.append('S')  # start token
smiles_vocab_target.append('E')  # end token
smiles_vocab_target.append('X')  # pad token
smiles_vocab_target.append('U')  # unknown token

tgt_vocab_size =len(smiles_vocab_target)
with open("data/smiles_con_vocab.pkl","wb") as f:
    pickle.dump(smiles_vocab_target,f)
f.close()

# 2. now produce smiles vocabulary based on atom tokenization
# and create vocabulary: "smiles_atom_vocab.pkl"
# uses: https://github.com/MolecularAI/pysmilesutils

smiles = df.canon_seq.tolist()
smiles = [b for b in smiles if isinstance(b, str)]
atom_tokenizer = SMILESAtomTokenizer(smiles=smiles)

smiles_vocab_target = list(atom_tokenizer.vocabulary.keys())
smiles_vocab_target = list(set(smiles_vocab_target))
smiles_vocab_target.remove(' ')
smiles_vocab_target.append('S')  # start token
smiles_vocab_target.append('E')  # end token
smiles_vocab_target.append('X')  # pad token
smiles_vocab_target.append('U')  # unknown token
fp = open("data/smiles_atom_vocab.pkl","wb")
pickle.dump(smiles_vocab_target,fp)