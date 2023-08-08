'''
Create pfam vocabulary and pickle it
Andrew Kiruluta, 05/22/2023
'''
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import pandas as pd
import pickle
import re

# load the data
seq = ""
df = pd.read_csv('data/df_inp1.csv')
for item in range(len(df)):
    try:
        seq += df.loc[item,"sequence"] + ' '
    except:
        pass
seq += 'nan'

pfam = list(set(seq.split(' ')))
file = open("data/pfam_vocab.pkl",'wb')
pickle.dump(pfam,file)
file.close()