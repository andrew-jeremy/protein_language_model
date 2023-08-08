'''
pytorch implementation of the transformer model for antiSMASH dataset
Andrew Kiruluta, 05/25/2023
'''
import torch
import torch.nn as nn
import torch.optim as optim
from time import sleep
import timeit
import shutil
import os, sys
import math
import glob
import tqdm
import config
import argparse
import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import train_test_split
from Transformer_torch_antismash import   TransformerCA_Antismash
from data_prep import data_loader
from pretrainer_dataset_antismash import ZincDataset, ZincDataset_loadall
from utils_antismash import get_protfam_smiles_encoding, random_word
from tqdm import tqdm
import numpy as np
#from scipy import stats
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import io
import matplotlib as mpl
mpl.use('Agg')  # No display
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import pickle
from torch.utils.tensorboard import SummaryWriter
fl = "pretrain_antismash_pfam_frozen_2"
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
    if ( lr < 1.0e-7 ):
        lr = 1.0e-7

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_valid_split(N):
    # Train/test split large dataset, downsample to N points
    # allocating train, test and validate datasets
    import random 
    try:
        os.remove('../output.txt')
    except OSError:
        pass

    with open('../zinc/combined_zinc_smiles.txt') as infile, open('../output.txt', 'a') as outfile: # random N sample from ZINC 880M
        outfile.writelines(random.sample(infile.readlines(), N))

    fin = open("../output.txt", 'rb') 
    f_train = open("data/train_zinc_smiles.txt", 'wb') 
    f_valid = open("data/valid_zinc_smiles.txt", 'wb')

    for line in fin: 
        r = random.random() 
        if (0.0 <=  r <= 0.85): 
            f_train.write(line) 
        else:
            f_valid.write(line)
    fin.close() 
    f_train.close() 
    f_valid.close() 
    return 0
def push_all_zeros_back(a):
    # Based on http://stackoverflow.com/a/42859463/3293881
    valid_mask = a!=0
    flipped_mask = valid_mask.sum(1,keepdims=1) > np.arange(a.shape[1]-1,-1,-1)
    flipped_mask = flipped_mask[:,::-1]
    a[flipped_mask] = a[valid_mask]
    a[~flipped_mask] = 0
    return a

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transformer model')
    parser.add_argument('--output_dir', type=str, default='data', help='Outfile directory')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--nlayers', type=int, default=1, help='Number of layers')
    parser.add_argument('--antismash', type=bool, default=True, help='True: with both pfam & antismash encoders, False: pfam encoder only')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--dataset_size', type=bool, default=False, help='calculate dataset size or not')
    parser.add_argument('--random_samples', type=bool, default=False, help='create new random train/test samples split or not from ZINC 880M')
    parser.add_argument('--shannon_samples', type=bool, default=True, help='create new train/test samples split or not from ZINC 880M using shannon entropy')
    parser.add_argument('--lr_decay', type=bool, default=True, help='whether to decay learning rate or not')
    parser.add_argument('--load_in_memory', type=bool, default=False, help='whether to load entire training in data in memory or read from disk (too large) True: memory, False:read from disk')
    parser.add_argument('--pretrained', type=bool, default=False, help='initialize model with pretrained weights')
    parser.add_argument('--train', type=int, default=1, help='whether to run inference or not 1:train, 2:inference')
    parser.add_argument('--train', type=int, default=1, help='whether to run inference or not 1:train, 2:inference')
    parser.add_argument('--mixed_precision', type=bool, default=True, help='FP16: True, FP32: False')
    
    args = parser.parse_args()
    src_vocab_size = config.src_vocab_size 
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
    if args.antismash: # with both pfsm & antismash encoders
        model = TransformerCA_Antismash(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    else:  # only pfam encoder, no antismash
        model = TransformerCA(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    
    if args.pretrained == True:  # load pretrained model
        model = torch.load('data/model_pretrain_antismash_fuse_atom.pth',map_location=torch.device('cuda')) 
            
    # set pfma cross attention head to zero weights
    for i in range(len(model.decoder_layers)):
        model.decoder_layers[i].cross_attn1.W_q.weight.data.fill_(0)
        model.decoder_layers[i].cross_attn1.W_k.weight.data.fill_(0)
        model.decoder_layers[i].cross_attn1.W_v.weight.data.fill_(0)
        model.decoder_layers[i].cross_attn1.W_o.weight.data.fill_(0)
    
    # then freeze pfam encoder pfam cross attention head in the decoder  
    modules_to_freeze = [model.encoder1_layers[i] for i in range(len(model.encoder1_layers))]
    modules_to_freeze.extend([model.decoder_layers[i].cross_attn1 for i in range(len(model.decoder_layers))])
    for module in modules_to_freeze:
        for param in module.parameters(): 
                param.requires_grad = False
                #print(param.requires_grad)

    # move to GPU/MPS
    if torch.backends.mps.is_available():
        device = 'mps'  # mac M1
        model = torch.nn.DataParallel(model).to(device)
        model = model.to(device)
        print('Using MPS')
    elif torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not args.pretrained: 
            model = torch.nn.DataParallel(model) # if pretrained, already mapped to multiple GPUs...
        model = model.to(device)              
        #model = torch.nn.parallel.DistributedDataParallel(model).to(device)  # multiple GPU, multiple nodes
        print("Using CUDA")
    else:
        device = 'cpu'
        model = model.to(device)
        print('Using CPU')

    use_amp = True  # use mixed precision
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = CosineAnnealingLR(optimizer,
                               T_max = max_epochs, # Maximum number of iterations.
                               eta_min = 1e-9)    # Minimum learning rate.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # load pfam vocabulary: generated vocab from "pfam_tokenizer_vocab.py"
    pickle_in = open("data/pfam_vocab.pkl","rb") 
    pfam_vocab = pickle.load(pickle_in)
    pickle_in.close()
    ptoi = {c: i for i, c in enumerate(pfam_vocab)}
    pfam_tokens = list(ptoi.values())
    
    # load smiles vocabulary: generated vocab from "smiles_tokenizer_vocab.py"
    fp = open("data/smiles_atom_vocab.pkl", "rb")
    smiles_vocab_target = pickle.load(fp)
    fp.close()
    print(f'Target vocab size: {len(smiles_vocab_target)}')
    
    smiles_vocab_target.insert(0,smiles_vocab_target.pop(smiles_vocab_target.index('X'))) # pad token
    stoi = {c: i for i, c in enumerate(smiles_vocab_target)}
    
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
    
    if args.train == 1: # training mode
        model.train()

        validation_split = .2
        shuffle_dataset = True
        random_seed= 42

        # determine size of ZINC dataset (880M) and create train/val split:
        if args.random_samples:
            with open("data/combined_zinc_smiles.txt") as fp:
                count = 0
                for _ in fp:
                    count += 1
            dataset_size = count
        else:
            dataset_size = 128 #883777337
        
        #  randomly sampled ZINC (880M)  1M train/valid datasets
        if args.random_samples:
            N=1000000 # 1M taining SMILES samples
            train_test_split(N)

        if args.shannon_samples: # sampled ZINC (880M) using shannon entropy to 10M train/valid datasets
            fin = open("data/zinc_sample.txt", 'rb') 
            f_train = open("data/train_zinc_smiles.txt", 'wb') 
            f_valid = open("data/valid_zinc_smiles.txt", 'wb')

            for line in fin: 
                r = random.random() 
                if (0.0 <=  r <= 0.85): 
                    f_train.write(line) 
                else:
                    f_valid.write(line)
            fin.close() 
            f_train.close() 
            f_valid.close()

        # load data
        if args.load_in_memory:
            df_train = pd.read_fwf('data/train_zinc_smiles.txt', names=["smiles"])
            df_valid = pd.read_fwf('data/valid_zinc_smiles.txt', names=["smiles"])
            train_dataset = ZincDataset_loadall(df_train)
            val_dataset   = ZincDataset_loadall(df_valid)
        else:
            train_dataset = ZincDataset('data/train_zinc_smiles.txt')
            val_dataset   = ZincDataset('data/valid_zinc_smiles.txt')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
        valid_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

        f1=open('data/train_loss.pkl', 'wb')
        f2=open('data/valid_loss.pkl', 'wb')
        
        best_acc1 = 99999
        
        for epoch in range(max_epochs): #tqdm(range(3)):
            print("Epoch: {}".format(epoch))
            train_loss = 0.0
            pkl_loss = []
            model.train()
            for batch_idx, data in enumerate(tqdm(train_loader)):
                sleep(0.01)

                start = timeit.default_timer()
                data = list(map(lambda x: x.strip(),data))             # input consist of a single smile string which is used as input to the dummy encoders & the decoder
                data = list(map(get_smiles_encoding,data))
                target = np.array(data)

                #  pfam encoder inputs
                #tgt_data_masked = list(map(partial(random_word,mask_prob=0.15),data))         # mask token with 15% probability -> NOTE USED FOR TRAINING
                #data_masked = np.array(tgt_data_masked)
                #src_data_1 = torch.from_numpy(np.array(tgt_data_masked)) 
                
                #  pfam encoder inputs  
                p = random.randint(0, 4)/10 # 0.0, 0.1, 0.2, 0.3, 0.4 masking probability
                src_data_1 = np.random.choice(pfam_tokens, size=(target.shape[0], target.shape[1]),replace=True)
                indices = np.random.choice(src_data_1.shape[1]*src_data_1.shape[0], replace=False, size=int(src_data_1.shape[1]*src_data_1.shape[0]*p))
                src_data_1[np.unravel_index(indices, src_data_1.shape)] = 0  # mask tokens for variable length pfams
                src_data_1 = torch.from_numpy(push_all_zeros_back(np.array(src_data_1)))  
                
                source_anti = list(map(partial(random_word,mask_prob=0.45),data))       # mask antismash encoder input tokens with 45% probability
                src_data_2 = torch.from_numpy(np.array(source_anti))    # to antismash encoder -FROZEN, INPUT IS IRRELEVANT
                
                tgt_data_real = torch.from_numpy(target)              # real targets - ground truth
                tgt_data_inp =  torch.from_numpy(target)              # decoder input - ground truth

                # move to GPU/MPS
                src_data_1 = src_data_1.to(device)
                src_data_2 = src_data_2.to(device)
                tgt_data_inp = tgt_data_inp.to(device)
                tgt_data_real = tgt_data_real.to(device)   

                optimizer.zero_grad()
                if args.mixed_precision: # FP16 precision
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        output = model(src_data_1, src_data_2, tgt_data_inp[:, :-1])
                        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data_real[:, 1:].contiguous().view(-1))
                    scaler.scale(loss).backward()
                    scaler.step(optimizer) #optimizer.step()
                    scaler.update()
                else: # FP32 precision
                    output = model(src_data_1, src_data_2, tgt_data_inp[:, :-1])
                    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data_real[:, 1:].contiguous().view(-1))
                    output = model(src_data_1, src_data_2, tgt_data_inp[:, :-1])
                    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data_real[:, 1:].contiguous().view(-1))
                    loss.backward()
                    optimizer.step()
                
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                train_loss += loss.item()
                pkl_loss.append(loss.item())
                
                # decay the learning rate based on our progress
                if args.lr_decay:
                    lr = adjust_learning_rate_poly(optimizer, args.learning_rate, epoch, max_epochs)
                    scheduler.step()
                    lr = optimizer.param_groups[0]["lr"]
                    #print(f"Learning rate: {lr}")
                stop = timeit.default_timer()
                #print('Run Time: ', stop - start)
            train_loss /= batch_idx
            writer.add_scalar("train loss", np.mean(train_loss), epoch)
            print(f"Epoch: {epoch}, Train Loss: {train_loss}")
            pickle.dump(pkl_loss, f1)
            print(f"Learning rate: {lr}")

            if epoch % 5 == 0:
                test_loss = 0.0
                pkl_loss = []
                model.eval()     # Optional when not using Model Specific layer
                for batch_idx, data in enumerate(tqdm(valid_loader)):
                    data = list(map(lambda x: x.strip(),data))
                    data = list(map(get_smiles_encoding,data))
                    target = np.array(data)
                    
                    tgt_data_masked = list(map(partial(random_word,mask_prob=0.15),data))         #  # mask token with 15% probability 
                    tgt_data_masked = np.array(tgt_data_masked)

                    # pfam encoder inputs
                    p = random.randint(0,4)/10 # 0.0, 0.1, 0.2, 0.3, 0.4 masking probability
                    src_data_1 = np.random.choice(pfam_tokens, size=(target.shape[0], target.shape[1]),replace=True)
                    indices = np.random.choice(src_data_1.shape[1]*src_data_1.shape[0], replace=False, size=int(src_data_1.shape[1]*src_data_1.shape[0]*p))
                    src_data_1[np.unravel_index(indices, src_data_1.shape)] = 0  # mask tokens for variable length pfams
                    src_data_1 = torch.from_numpy(push_all_zeros_back(np.array(src_data_1)))       
                
                    source_anti = list(map(partial(random_word,mask_prob=0.45),data))       # mask antismash encoder input tokens with 45% probability
                    src_data_2 = torch.from_numpy(np.array(source_anti))    # to antismash encoder, frozen too, does not matter
                
                    #tgt_data_inp = torch.from_numpy(tgt_data_masked)     # randomly masked decoder input
                    tgt_data_real = torch.from_numpy(target)             # real targets - ground truth
                    tgt_data_inp = torch.from_numpy(target) 
                    
                    # move to GPU/MPS
                    src_data_1 = src_data_1.to(device)
                    src_data_2 = src_data_2.to(device)
                    tgt_data_inp = tgt_data_inp.to(device)
                    tgt_data_real = tgt_data_real.to(device)
                
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        output = model(src_data_2, src_data_2, tgt_data_inp[:, :-1])
                        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data_real[:, 1:].contiguous().view(-1))
                    
                    if args.mixed_precision: # FP16 precision
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                            output = model(src_data_1, src_data_2, tgt_data_inp[:, :-1])
                            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data_real[:, 1:].contiguous().view(-1))
                        scaler.scale(loss).backward()
                        scaler.step(optimizer) #optimizer.step()
                        scaler.update()
                    else: # FP32 precision
                        output = model(src_data_1, src_data_2, tgt_data_inp[:, :-1])
                        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data_real[:, 1:].contiguous().view(-1))
                        output = model(src_data_1, src_data_2, tgt_data_inp[:, :-1])
                        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data_real[:, 1:].contiguous().view(-1))
                        loss.backward()
                        optimizer.step()
                       
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    test_loss += loss.item()
                    pkl_loss.append(loss.item())

                test_loss /= batch_idx
                writer.add_scalar("valid loss", np.mean(test_loss), epoch)
                print(f"Epoch: {epoch}, Valid Loss: {test_loss}")
                pickle.dump(pkl_loss, f2)

                # remember best best_acc1 and save checkpoint
                if np.mean(test_loss) < best_acc1:
                    best_acc1 = np.mean(test_loss)
                    torch.save(model, 'data/checkpoint_atom_10M.pth')
                        
            if epoch % 10 == 0:
                torch.save(model, 'data/model_'+fl+'.pth')
                fp.close()   
        f1.close()
        f2.close()