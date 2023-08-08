'''
Custom tokenization for SMILES and protein family sequences for antiSMASH data
Andrew Kiruluta, 05/22/2023
'''
import re
import numpy as np
import torch
import random
from tokenizers import Tokenizer
import pickle
import selfies as sf
import config

MAX_PROT_LEN =  config.max_seq_length 
MAX_SMILES_LEN = config.max_seq_length 

MAX_AA = 5

# AA_vocab and pos_vocab were used to train activity oracle
AA_map = {'<PAD>': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4,
            'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11,'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}


smiles_alphabet = {'[start]','#', '(', ')', '-', '.', '=',
                   '1', '2', '3', '4', '5', '6', '7', '8',
                   'Br', 'C', 'l', 'F', 'I', 'N', 'O', 'P', 'S',
                   '[B-]', '[Br-]', '[H]', '[K]', '[Li]', '[N+]',
                   '[N-]', '[NH+]', '[NH2+]', '[NH3+]', '[Na+]','[*]',
                   '[O-]', '[OH-]', '[P-]', '[Pt+2]', '[Pt]', '[S+]',
                   '[SH]', '[Si]', '[n+]', '[nH+]', '[nH]',
                   'c', 'n', 'o', 's','[end]'}

# atom-level tokens used for trained the spe vocabulary
# atom_toks
atom_toks = {'[c-]', '[SeH]', '[N]', '[C@@]', '[Te]', '[OH+]', 'n', '[AsH]', '[B]', 'b', 
             '[S@@]', 'o', ')', '[NH+]', '[SH]', 'O', 'I', '[C@]', '-', '[As+]', '[Cl+2]', 
             '[P+]', '[o+]', '[C]','Cl', '[C@H]', '[CH2]', '\\', 'P', '[O-]', '[NH-]', '[S@@+]', 
             '[te]', '[s+]', 's', '[B-]', 'B', 'F', '=', '[te+]', '[H]', '[C@@H]', '[Na]', 
             '[Si]', '[CH2-]', '[S@+]', 'C', '[se+]', '[cH-]', '6', 'N', '[IH2]', '[As]', 
             '[Si@]', '[BH3-]', '[Se]', 'Br', '[C+]', '[I+3]', '[b-]', '[P@+]', '[SH2]', '[I+2]', 
             '%11', '[Ag-3]', '[O]', '9', 'c', '[N-]', '[BH-]', '4', '[N@+]', '[SiH]', '[Cl+3]', '#', 
             '(', '[O+]', '[S-]', '[Br+2]', '[nH]', '[N+]', '[n-]', '3', '[Se+]', '[P@@]', '[Zn]', '2', 
             '[NH2+]', '%10', '[SiH2]', '[nH+]', '[Si@@]', '[P@@+]', '/', '1', '[c+]', '[S@]', '[S+]', 
             '[SH+]', '[B@@-]', '8', '[B@-]', '[C-]', '7', '[P@]', '[se]', 'S', '[n+]', '[PH]', '[I+]', '5', 'p', '[BH2-]', '[N@@+]', '[CH]', 'Cl'}


smiles_regex = re.compile(
    r'(\[[^\]]+]|C|l|Cr|Br|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|[start]|[end]|'
    r'-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
)

SMILES_map = {'<PAD>': 0}
SMILES_map['<mask>'] = 1
for idx, char in enumerate(smiles_alphabet, start = 2):
#for idx, char in enumerate(atom_toks, start = 2):
    SMILES_map[char] = idx + 1
SMILES_map['<Unk>'] = len(SMILES_map)

SMILES_vocab = {v: k for k, v in SMILES_map.items()}
fp = open("data/smiles_vocab.pkl","wb")
pickle.dump(SMILES_vocab,fp)
fp.close()

inv_smiles_vocab = {v: k for k, v in SMILES_vocab.items()}

def get_smiles_encoding(smiles):
    encoding = [0]*MAX_SMILES_LEN
    if len(smiles) > MAX_SMILES_LEN:
        smiles = smiles[:MAX_SMILES_LEN]
    for i, char in enumerate(smiles_regex.split(smiles)[1::2]):
        encoding[i] = SMILES_map.get(char, SMILES_map['<Unk>']) # 0 is reserved for padding
    return encoding

def selfie_alphabet_builder(smiles):
    alphabet=sf.get_semantic_robust_alphabet() # Gets the alphabet of robust symbols
    alphabet.add("[nop]")  # [nop] is a special padding symbol
    alphabet.add('[start]')
    alphabet.add('[end]')
    s = smiles.replace('([*])','')
    if len(s) == 0:
        selfies = '[nop]'
    else:
        try:
            selfies = sf.encoder(s)
        except:
            selfies = '[nop]'
    return selfies

# TBD: selfies tokenization
# https://github.com/aspuru-guzik-group/selfies
def get_selfies_encoding(smiles,dataset):
    # convert smiles to selfies
    # SMILES -> SELFIES -> translation
    #alphabet=sf.get_semantic_robust_alphabet() # Gets the alphabet of robust symbols
    alphabet = sf.get_alphabet_from_selfies(dataset)
    alphabet.add("[nop]")  # [nop] is a special padding symbol
    alphabet.add('.')
    alphabet.add('[start]')
    alphabet.add('[end]')
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
    #try:
    s = smiles.replace('([*])','')
    
    if len(s) == 0:
        selfies = '[nop]'
    else:
        if '[start]' not in s: # no start token => not target
            selfies = sf.encoder(s)
        else:   # target
            s = s.replace('[start]','')
            s = s.replace('[end]','')
            try:
                selfies = sf.encoder(s)
            except:
                selfies = '[nop]'
            selfies = '[start]' + selfies + '[end]'

    encoding, _ = sf.selfies_to_encoding(selfies, symbol_to_idx, pad_to_len=MAX_SMILES_LEN, enc_type="both")
    return encoding[:MAX_SMILES_LEN]

def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0]*(target_len - len(some_list))


# smiles tokenization using ProtFam encoding with TfidfVectorizer()
def get_protfam_smiles_encoding(seq, vocab):
    seq = seq.lower().split() 
    encoding = [0]*MAX_PROT_LEN
    if len(seq) > MAX_PROT_LEN:
        seq = seq[:MAX_PROT_LEN]
    for i,item in enumerate(seq):
        encoding[i] = vocab[item]
    return encoding 


def random_word(tokens,mask_prob):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of int, tokenized smiles.
    :param tokenizer: inv_smiles_vocab, object used for tokenization (we need it's vocab here)
    :return: (list of int), masked tokens and related labels for LM prediction
    -100 for unmasked tokens, their loss will be later ignored in CrossEntropyLoss calculation.
    """

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with mask_prob % probability
        if prob < mask_prob:
            prob /= mask_prob

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = SMILES_map['<mask>']

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = inv_smiles_vocab[random.choice(list(SMILES_vocab.values()))]

            # -> rest 10% randomly keep current token
            else:
                tokens[i] = token
        else:
            # -100 is the default value that gets ignored by the PyTorch CrossEntropyLoss method. 
            # When doing masked language modeling, we only compute the loss on the masked tokens.
            # ignore_index (int, optional) 
            tokens[i] = 0

    return tokens # source


# protein family tokenization using ProtFam pretrained-model
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)