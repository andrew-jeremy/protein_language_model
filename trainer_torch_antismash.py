'''
pytorch implementation of the transformer model for antiSMASH dataset
Andrew Kiruluta, 05/25/2023
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import shutil
import os, sys
import math
import glob
import config
import tqdm
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Transformer_torch_antismash import  TransformerCA, TransformerCA_Antismash
from data_prep import  data_loader
from dataset_antismash import AntiSmashDataset
from Optim import ScheduledOptim
from tokenizers import Tokenizer
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from pysmilesutils.tokenize import *
from pysmilesutils.analyze import analyze_smiles_tokens
import torch
import random
import io
import matplotlib as mpl
mpl.use('Agg')  # No display
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from pytorch_beam_search import seq2seq
import pickle
from fuzzywuzzy import fuzz
import  Levenshtein

from torch.utils.tensorboard import SummaryWriter
fl = "transformer_antismash_fuse_pretrain_pfam_fc_lr"
writer = SummaryWriter('run/'+fl)

import warnings
warnings.filterwarnings("ignore")

class ScatterPlots:
    def __init__(self, r, mae, y):
        self.r = r
        self.mae = mae
        self.y = y

    def plot2(self,fig):
        with io.BytesIO() as buff:
            fig.savefig(buff, format='png')
            buff.seek(0)
            im = plt.imread(buff)
            return im

    def tensorboard_plot(self,img_batch,epoch):
        img_batch = img_batch[:, :, :3]      # convert to 3 channels
        img_batch = np.transpose(img_batch, [2, 0, 1])
        img_batch = img_batch[None,:, :, :] # convert to 1 batch (None), replace with batch_size here...
        writer.add_images('image_batch', img_batch, 0)
        writer.add_figure('epoch_{:d}'.format(epoch), plt.gcf(), 0)
        writer.close()

def adjust_learning_rate_poly(optimizer, initial_lr, iteration, max_iter):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = initial_lr * ( 1 - (iteration / max_iter)) * ( 1 - (iteration / max_iter))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

        
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return 
    
# greedy search implementation for inference
def decode_sequence(input_sentence_1, input_sentence_2, model, max_length, device='cuda', pfam_vocab=None, smiles_vocab=None):
    tokenized_input_sentence_1 = get_protfam_smiles_encoding(input_sentence_1) 
    tokenized_input_sentence_1 = torch.tensor(tokenized_input_sentence_1,dtype=torch.long).to(device)
    tokenized_input_sentence_2 = get_smiles_encoding(input_sentence_2)
    tokenized_input_sentence_2 = torch.tensor(tokenized_input_sentence_2,dtype=torch.long).to(device)

    decoded_sentence = "S"
    tok = []
    for i in range(max_length):
        tokenized_target_sentence = get_smiles_encoding(decoded_sentence) 
        tokenized_target_sentence = torch.tensor(tokenized_target_sentence,dtype=torch.long).to(device)

        # predictions for each token at each timestep of vocab size
        predictions = model(tokenized_input_sentence_1.unsqueeze(0), tokenized_input_sentence_2.unsqueeze(0), tokenized_target_sentence.unsqueeze(0)[:,:-1])
        predictions = predictions.cpu().detach().numpy()
        sampled_token_index = np.argmax(predictions[0, i, :]) # greedy sampling picking the highest probability token at each timestep
        tok.append(sampled_token_index)       
        try:
            sampled_token = itos[sampled_token_index]
        except:
            sampled_token = "U"
        decoded_sentence += sampled_token

        if sampled_token == "E":
            break
    print(f"predicted: {decoded_sentence}")
    #print(f"token indices: {tok}")
    print(f"# of unique tokens: {set(tok)}")
    return decoded_sentence

# beam search decoding algorithm for inference
def beam_search(input_sentence_1, input_sentence_2, model, device='cuda'):
    tokenized_input_sentence_1 = get_protfam_smiles_encoding(input_sentence_1) 
    tokenized_input_sentence_1 = torch.tensor(tokenized_input_sentence_1,dtype=torch.long).to(device)
    tokenized_input_sentence_2 = get_smiles_encoding(input_sentence_2)
    tokenized_input_sentence_2 = torch.tensor(tokenized_input_sentence_2,dtype=torch.long).to(device)

    decoded_sentence = "S"
    tokenized_target_sentence = get_smiles_encoding(decoded_sentence) 
    tokenized_target_sentence = torch.tensor(tokenized_target_sentence,dtype=torch.long).to(device)
    
    # predictions for each token at each timestep of vocab size
    predictions = model(tokenized_input_sentence_1.unsqueeze(0), tokenized_input_sentence_2.unsqueeze(0), tokenized_target_sentence.unsqueeze(0)[:,:-1])
    predictions = predictions.cpu().detach().numpy()
    return predictions

# beam search decoder
# Instead of greedily choosing the most likely next step as the sequence is constructed, 
# the beam search expands all possible next steps and keeps the k most likely, where k is a 
# user-specified parameter and controls the number of beams or parallel searches through the 
# sequence of probabilities.
# Common beam width values are 1 for a greedy search and values of 5 or 10 for common benchmark 
# problems in machine translation. Larger beam widths result in better performance of a model as 
# the multiple candidate sequences increase the likelihood of better matching a target sequence. 
# This increased performance results in a decrease in decoding speed.
def beam_search_decoder(data, k):
    # first convert logits to probabilites so that all numbers are +ve
    data = torch.softmax(data, dim=2).numpy()[0,:,:]
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
#             for j in range(len(row)): # instead of exploring all the labels, explore only k best at the current time
            # select k best
            best_k = np.argsort(row)[-k:]
            # explore k best
            for j in best_k:
                candidate = [seq + [j], score + math.log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)
        # select k best
        sequences = ordered[:k]
    
    decoded = 'S'
    for item in sequences[k-1][0]: # pick one with the highest log likelihood probability
        decoded += itos[item]
        if itos[item] == 'E':
            break
    return decoded

def search(fullstring, substring):
    def check(s1, s2):
        for a, b in zip(s1, s2):
            if a != b and b != "*":
                return False
        return True

    for i in range(len(fullstring) - len(substring) + 1):
        if check(fullstring[i : i + len(substring)], substring):
            return True

    return False

def model_layer_state(model,layer_to_skip, trainable=False):
    for name, param in model.named_parameters():
        if search(name,layer_to_skip): # skip some specific layers
            pass
        else:
            param.requires_grad = trainable

# canonize smiles
def canonize(x):
    if isinstance(x, str) and x.lower() == "nan":
        return x
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(x))
    except:
        return x
        
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transformer model')
    parser.add_argument('--output_dir', type=str, default='data', help='Outfile directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--nlayers', type=int, default=1, help='Number of layers')
    parser.add_argument('--antismash', type=bool, default=True, help='True: with antismash encoder False: no antismash encoder')
    parser.add_argument('--tokenizer', type=str, default='atom', help='smiles tokenizer type: atom or char')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--lr_decay', type=bool, default=True, help='whether to decay learning rate or not')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--train', type=bool, default=True, help='whether to run inference or not\
                        True:train, False:inference')
    parser.add_argument('--pretrained', type=bool, default=True, help='initialize model with pretrained weights')
    parser.add_argument('--decoder_type', type=int, default=0, help='0: greedy, 1: beam search')
    parser.add_argument('--load_data', type=int, default=1, help='0: mibig_2.0, 1: mibig_3.1') 
    
    args = parser.parse_args()

    src_vocab_size = config.src_vocab_size # encoder pfam vocab - 3655 actual
    tgt_vocab_size = config.tgt_vocab_size 
    d_model = config.d_model 
    num_heads = config.num_heads 
    num_layers = config.num_layers 
    d_ff = config.d_ff     
    max_seq_length = config.max_seq_length  
    dropout = args.dropout
    output_dir = args.output_dir
    batch_size = args.batch_size
    nlayers = args.nlayers
    max_epochs = args.epochs
    seed = args.seed
    
    # Training Model
    if args.antismash: # with antismash encoder
        model = TransformerCA_Antismash(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    else:  # only pfam encoder, no antismash
        model = TransformerCA(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
        
    # load pretrained model (ZINC dataset) if specified
    if args.pretrained == True:
        if torch.backends.mps.is_available(): # using mps...
            model = torch.load('data/checkpoint_pretrain_antismash_pfam_frozen_epoch_65.pth', map_location=torch.device('mps'))  
            
            # layers to freeze: decoder, encoder2 (antismash), decoder feedforward
            modules_to_freeze = [model.module.decoder_layers[i] for i in range(len(model.module.decoder_layers))]
            #modules_to_freeze.extend([model.module.encoder2_layers[i] for i in range(len(model.module.encoder2_layers))])
            #modules_to_freeze.extend([model.module.fc]) 
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad =  False
            
            # layers to train: pfam encoder &  pfam cross attention head in the decoder  
            model_layer_state(model.module.encoder1_layers,layer_to_skip='norm',trainable=True)
            modules_to_train = [model.module.decoder_layers[i].cross_attn1 for i in range(len(model.module.decoder_layers))]
            modules_to_train.extend([model.module.fc])
            #modules_to_train.extend([model.module.decoder_layers[i].cross_attn2 for i in range(len(model.module.decoder_layers))])
            for module in modules_to_train:
                for param in module.parameters():
                    param.requires_grad = True
            # now initialize pfam encoder cross attention weights
            for i in range(len(model.module.decoder_layers)): # initialize pfam cross attention weights 
                torch.nn.init.xavier_uniform(model.module.decoder_layers[i].cross_attn1.W_q.weight)
                torch.nn.init.xavier_uniform(model.module.decoder_layers[i].cross_attn1.W_k.weight)
                torch.nn.init.xavier_uniform(model.module.decoder_layers[i].cross_attn1.W_v.weight)
                torch.nn.init.xavier_uniform(model.module.decoder_layers[i].cross_attn1.W_o.weight)
                model.module.decoder_layers[i].cross_attn1.W_q.bias.data.fill_(0.01)
                model.module.decoder_layers[i].cross_attn1.W_k.bias.data.fill_(0.01)
                model.module.decoder_layers[i].cross_attn1.W_v.bias.data.fill_(0.01)
                model.module.decoder_layers[i].cross_attn1.W_o.bias.data.fill_(0.01)
                
            #model_layer_state(model.module.decoder_layers,layer_to_skip='norm',trainable=False)  # decoder       
            #model_layer_state(model.module.fc,layer_to_skip='',trainable=False)                  # fc
            #model_layer_state(model.module.encoder2_layers,layer_to_skip='norm',trainable=False) # antismash encoder
            #model_layer_state(model.module.encoder1_layers,layer_to_skip='norm',trainable=True)  # pfam encoder
                                             
        else: # using cuda...
            model = torch.load('data/checkpoint_atom_pretrain.pth', map_location=torch.device('cuda'))   
            # layers to freeze: decoder, encoder2 (antismash), decoder feedforward
            modules_to_freeze = [model.decoder_layers[i] for i in range(len(model.decoder_layers))]
            modules_to_freeze.extend([model.encoder2_layers[i] for i in range(len(model.encoder2_layers))])
            modules_to_freeze.extend([model.fc]) 
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad =  False
            
            # layers to train: pfam encoder &  pfam cross attention head in the decoder  
            modules_to_train = [model.encoder1_layers[i] for i in range(len(model.encoder1_layers))]
            modules_to_train.extend([model.decoder_layers[i].cross_attn1 for i in range(len(model.decoder_layers))])
            for module in modules_to_freeze:
                for param in module.parameters(): 
                    param.requires_grad = True
           
    # The log-softmax function will not be applied in the transformer decoder due to the  use of CrossEntropyLoss here, 
    # which requires the inputs to be unnormalized logits.              
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) #,momentum=args.momentum,weight_decay=args.weight_decay)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    #scheduler = CosineAnnealingLR(optimizer,
    #                          T_max = max_epochs, # Maximum number of iterations.
    #                           eta_min = 1e-9)     # Minimum learning rate.
    scheduler  = ScheduledOptim(optimizer, lr_mul=1e-3, d_model=d_model, n_warmup_steps=5)
    
    # now move to GPU/MPS
    if torch.backends.mps.is_available():
        device = 'mps'
        if args.pretrained == True:
            model = model.module.to(device)
        else:
            model = model.to(device)
        print('Using MPS')
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model).to(device) # multiple GPUs     
        #model = torch.nn.parallel.DistributedDataParallel(model).to(device)  # multiple GPU, multiple nodes
        print("Using CUDA")
    else:
        device = 'cpu'
        print('Using CPU')
        
    '''
    # debugging point - put everthing on cpu for more detailed errors and comment out section above
    device = torch.device("cpu")
    model = model.to(device)
    '''
    
    # load pretrained model
    #model  = torch.load('data/pretrain_model.pth',map_location=torch.device('mps'))
    
    # load pfam vocabulary: generated vocab from "pfam_tokenizer_vocab.py"
    pickle_in = open("data/pfam_vocab.pkl","rb") 
    pfam_vocab = pickle.load(pickle_in)
    pickle_in.close()
    ptoi = {c: i for i, c in enumerate(pfam_vocab)}
    
    if args.tokenizer == 'atom':
        # atom smiles tokenizer vocabularly for smiles
        # uses: https://github.com/MolecularAI/pysmilesutils
        fp = open("data/smiles_atom_vocab.pkl","rb")      
    else:
        # load smiles char tokenizer vocabulary: generated vocab from "smiles_tokenizer_vocab.py"
        fp = open("data/smiles_con_vocab.pkl","rb") 
        
    smiles_vocab_target = pickle.load(fp) 
    smiles_vocab_target.insert(0,smiles_vocab_target.pop(smiles_vocab_target.index('X'))) # put pad index at beginning of list 
    stoi = {c: i for i, c in enumerate(smiles_vocab_target)}
    itos = {i: c for i, c in enumerate(smiles_vocab_target)}
    fp.close()
    
    # load smiles vocabulary: generated vocab from "smiles_tokenizer_vocab.py"
    def get_smiles_encoding(s):
        encoding = [0]*config.max_seq_length
        if len(list(s)) > config.max_seq_length:
            s = s[:config.max_seq_length]
        for i,j in enumerate(s):
            try:
                encoding[i] = stoi[j] # encode known smile character  to integer
            except:
                encoding[i] = stoi['U'] # new previously unseen character mapped to 'U' token
        return encoding 
    
    def get_protfam_smiles_encoding(s):
        encoding = [0]*config.max_seq_length
        if len(list(s)) > config.max_seq_length:
            s = s[:config.max_seq_length]
        for i,j in enumerate(s.split(' ')):
            try:
                encoding[i] = ptoi[j] # encode known pfam  to integer
            except:
                continue
        return encoding 
    
    if args.train: # training mode
        model.train()
        # load data mibig_2.0_metadata_pfam_df.txt dataset
        if args.load_data == 0:
            filename = "data/mibig_2.0_metadata_pfam_w_antismash_struct_df.txt"
            inp1, inp2, targ = data_loader(filename)
        else: # load data mibig_3.1_metadata_pfam_df.txt dataset
            filename = "data/mibig_3.1/mibig_3.1_metadata_pfam_w_antismash_struct_df.txt"
            inp1, inp2, targ = data_loader(filename)

        df_inp1 = pd.DataFrame(np.array(inp1),columns=['pfam']) # pfam
        df_inp2 = pd.DataFrame(np.array(inp2),columns=['antismash']) # smiles
        df_targ = pd.DataFrame(np.array(targ),columns=['target']) # smiles
        df_inp1.to_csv('data/df_inp1.csv') # df_inp1.pfam.str.len().min() = 7 df_inp1.pfam.str.len().max() = 871
        df_inp2.to_csv('data/df_inp2.csv') # df_inp2.antismash.str.len().min() = 8 df_inp2.antismash.str.len().max() = 405
        df_targ.to_csv('data/df_targ.csv')
        
        # convert smiles to canonical smiles
        df_inp2['canon_seq2'] = df_inp2['antismash'].apply(lambda x: canonize(x)) # antiSMASH
        df_targ['canon_targ'] = df_targ['target'].apply(lambda x: canonize(x))
            
        df_inp1['tokens_1'] = df_inp1['pfam'].apply(lambda x: get_protfam_smiles_encoding(x)) # pfam
     
        df_inp2['tokens_2'] = df_inp2['canon_seq2'].apply(lambda x: get_smiles_encoding(x))    # antismash
        df_targ['tokens'] = df_targ['canon_targ'].apply(lambda x: get_smiles_encoding('S' + x + 'E')) # add start and end tokens only to targets
      
        df_src = pd.concat([df_inp1,df_inp2],axis=1)
        df_src = df_src[df_src['tokens_2'].notna()]
        
        X_train, X_test, y_train_targ, y_test_targ = train_test_split(df_src, df_targ, test_size=0.33, random_state=42)
        
        # save the SMILES_map for inference map
        fp = open("data/test_data.pkl","wb")
        pickle.dump([X_test, y_test_targ, X_train, y_train_targ],fp)
        fp.close()
        
        # reset index for each dataframe for Dataloader idx lookup
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True) 
        y_train_targ.reset_index(drop=True, inplace=True)
        y_test_targ.reset_index(drop=True, inplace=True)


        # create torch dataset
        train_dataset_inp = AntiSmashDataset(X_train,y_train_targ)
        test_dataset_inp = AntiSmashDataset(X_test,y_test_targ)

        trainloader = DataLoader(train_dataset_inp, shuffle=False, pin_memory=True,
                                    batch_size=args.batch_size,
                                    num_workers=1)

        validloader = DataLoader(test_dataset_inp, shuffle=False, pin_memory=True,
                                    batch_size=args.batch_size,
                                    num_workers=1)
        test_loss=0.0
        best_acc1 = 99999
        done = False
        
        #------DELETE LATER----->
        # too specific, will need to generalize
        test_pairs = list(zip(list(X_test.pfam),list(X_test.antismash), y_test_targ.iloc[:,0]))
        train_pairs = list(zip(list(X_train.pfam),list(X_train.antismash), y_train_targ.iloc[:,0]))
        random.shuffle(test_pairs)
        test_pairs = test_pairs[:5]

        if args.decoder_type == 0:  # greedy search algorithm
            for n in range(5):
                pfam, antismash, smiles = random.choice(train_pairs)
                pfam2, antismash2, smiles2 = random.choice(test_pairs)
                if smiles != 'nan' and smiles2 != 'nan':
                    break
        #------------------------------->
        
        for epoch in range(max_epochs): #tqdm(range(3)):
            train_loss = 0.0
            model.train()
            for _, d_it in tqdm(enumerate(trainloader), total=len(trainloader)):
                src_data_1,src_data_2, tgt_data = d_it
                src_data_1 = src_data_1.to(device)
                src_data_2 = src_data_2.to(device)
                tgt_data = tgt_data.to(device)

                # decay the learning rate based on our progress
                optimizer.zero_grad()
                if args.lr_decay:
                    #lr = adjust_learning_rate_poly(optimizer, args.learning_rate, epoch, max_epochs)
                    output = model(src_data_1, src_data_2, tgt_data[:, :-1])
                    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
                    loss.backward()
                    #scheduler.step()
                    scheduler.step_and_update_lr()
                else:
                    output = model(src_data_1, src_data_2, tgt_data[:, :-1])
                    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
                    loss.backward()
                    optimizer.step()
                    
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                train_loss += loss.item()
                   
                if args.pretrained == True:
                    if np.mean(train_loss) < 0.5 and done == False:  
                    #if lr < 5e-7 and done == False:  
                        print(f"Learning rate: {lr}")
                        if torch.backends.mps.is_available(): # using Mac...
                            #model_layer_state(model.decoder_layers,layer_to_skip='norm',trainable=True)  # full decoder
                            modules_to_train = [model.decoder_layers[i].cross_attn2 for i in range(len(model.decoder_layers))]
                            modules_to_train.extend([model.decoder_layers[i].cross_attn1 for i in range(len(model.decoder_layers))])
                            modules_to_train.extend([model.encoder2_layers[i] for i in range(len(model.encoder2_layers))]) # antismash encoder
                            modules_to_train.extend([model.fc])
                            for module in modules_to_train:
                                for param in module.parameters():
                                    param.requires_grad = True   
                            
                            #model_layer_state(model.fc,layer_to_skip='',trainable=True)                  # fc
                            #model_layer_state(model.encoder2_layers,layer_to_skip='norm',trainable=True) # antismash encoder
                        else:
                            #model_layer_state(model.decoder_layers,layer_to_skip='norm',trainable=True)  
                            model_layer_state(model.fc,layer_to_skip='',trainable=True)                 
                            model_layer_state(model.encoder2_layers,layer_to_skip='norm',trainable=True) 
                        done = True
                     
            train_loss = train_loss / len(trainloader)
            writer.add_scalar("train loss", np.mean(train_loss), epoch)
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss}")
            
            # DEBUG POINT - TO BE DELETED ----->
            if args.decoder_type == 0:
                decoded = decode_sequence(pfam, antismash, model,config.max_seq_length-1,device,pfam_vocab,smiles_vocab_target)
                decoded2 = decode_sequence(pfam2, antismash2, model,config.max_seq_length-1,device,pfam_vocab,smiles_vocab_target)
            else:
                data = beam_search(pfam, antismash, model, device)
                data = torch.from_numpy(data).type(torch.FloatTensor)
                decoded = beam_search_decoder(data, k=5)
                
            print(f"ground truth train: {smiles}") 
            print(f"predicted train: {decoded}") 
            print(f"Levenshtein index train: {np.around(Levenshtein.ratio(decoded,smiles),2)}")
            # string similarity with Levenshtein Distance: The Levenshtein distance is the algorithm. 
            # It calculates the minimum number of operations you must do to change 1 string into another. 
            # The fewer changes means the strings are more similar. Higher ratio means better matching
            print("\n")
            print(f"ground truth test: {smiles2}") 
            print(f"predicted test: {decoded2}")
            print(f"Levenshtein index test: {np.around(Levenshtein.ratio(decoded2,smiles2),2)}")
            #print(f"similarity index: {SequenceMatcher(None, translated, smiles).ratio()}")
            #---->
            if epoch % 5 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"Learning rate: {lr}")
                valid_loss = 0.0
                model.eval()     # Optional when not using Model Specific layer
                for _, d_it in tqdm(enumerate(validloader), total=len(validloader)):
                    src_data_1, src_data_2, tgt_data = d_it
                    src_data_1 = src_data_1.to(device)
                    src_data_2 = src_data_2.to(device)
                    tgt_data = tgt_data.to(device)
            
                    output = model(src_data_1, src_data_2, tgt_data[:, :-1])
                    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    valid_loss += loss.item()
                valid_loss = valid_loss / len(validloader)
                writer.add_scalar("valid loss", np.mean(valid_loss), epoch)
                print(f"Epoch: {epoch+1}, Valid Loss: {valid_loss}")
            
            # remember best best_acc1 and save checkpoint
            if np.mean(valid_loss) < best_acc1:
                best_acc1 = np.mean(valid_loss)
                torch.save(model, 'data/checkpoint_pretrain_cross2.pth')
                fp = open("data/checkpoint_pretrain_cross2__val_loss.pkl","wb")
                pickle.dump(model,fp)
                fp.close()
                      
            if epoch % 10 == 0:   
                #torch.save({'epoch': epoch,
                #    'model_state_dict': model.state_dict(),
                #    'optimizer_state_dict': optimizer.state_dict(),
                #    'loss': train_loss}, 
	            #    'data/model.pth')
                torch.save(model, 'data/model_'+fl+'.pth')
                fp = open("data/checkpoint_pretrain_cross2_lr.pkl","wb")
                pickle.dump(model,fp)
                fp.close()
        
    else: # inference
        print("inference mode")
        #fp = open("data/checkpoint_train.pkl","rb")
        #model = pickle.load(fp)
        #fp.close()
        #model = torch.load('data/model_transformer_antismash_fuse_pretrain_cross2_lr6.pth')
        model = torch.load('data/checkpoint_pretrain_cross2.pth')
        model.eval()
        model.to(device)

        # load smiles vocab mapping
        if args.tokenizer == 'atom':
             fp = open("data/smiles_atom_vocab.pkl","rb")  
        else:
            fp = open("data/smiles_con_vocab.pkl","rb")
        smiles_vocab_target = pickle.load(fp)
        fp.close()
        
        # load test data for inference
        fp = open("data/test_data.pkl","rb")
        X_test, y_test_targ, X_train,y_train_targ = pickle.load(fp)
        fp.close()

        # get random test sentences from test set
        test_count = 5

        # too specific, will need to generalize
        #test_pairs = list(zip(list(X_test.sequence.iloc[:,0]),list(X_test.sequence.iloc[:,1]), y_test_targ.iloc[:,0]))
        #test_pairs = list(zip(list(X_train.sequence.iloc[:,0]),list(X_train.sequence.iloc[:,1]), y_train_targ.iloc[:,0]))
        test_pairs = list(zip(list(X_train.pfam),list(X_train.antismash), y_train_targ.iloc[:,0]))
        random.shuffle(test_pairs)
        test_pairs = test_pairs[:test_count]

        if args.decoder_type == 0:  # greedy search algorithm
            for n in range(test_count):
                pfam, antismash, smiles = random.choice(test_pairs)
                #smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles),True) # convert to cannonical smiles
                #antismash = Chem.MolToSmiles(Chem.MolFromSmiles(antismash),True)
                translated = decode_sequence(pfam, antismash, model,config.max_seq_length-1,device,pfam_vocab,smiles_vocab_target)
                print(f"Test {n}:")
                print(f"pfam: {pfam}\n")
                print(f"antismash: {antismash}\n")
                print(f"ground truth: {smiles}\n")
                print(f"-> {translated}")
                print(f"similarity index: {np.around(Levenshtein.ratio(translated,smiles),2)}") # % matching: p = (1 - l/m) × 100 
                print()

        else: # beam search algorithm
            for n in range(test_count):
                pfam, antismash, smiles= random.choice(test_pairs)
                if smiles == 'nan':
                    continue
                data = beam_search(pfam, antismash, model, device)
                data = torch.from_numpy(data).type(torch.FloatTensor)
                
                # decode sequence
                # the beam search expands all possible next steps and keeps 
                # the k most likely, where k is a user-specified parameter and 
                # controls the number of beams or parallel searches through the 
                # sequence of probabilities.
                # Common beam width values are 1 for a greedy search and values of 
                # 5 or 10 for common benchmark problems in machine translation. Larger 
                # beam widths result in better performance of a model as the multiple 
                # candidate sequences increase the likelihood of better matching a target 
                # sequence. This increased performance results in a decrease in decoding speed.
                # 
                k = 3 
                decoded = beam_search_decoder(data, k)
                print(f"-> {decoded}\n")
                print(f"ground truth: {smiles}\n")
                print(f"similarity index: {np.round(Levenshtein.ratio(decoded,smiles),decimal=2)}")
                        