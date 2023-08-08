'''
Transformer model helper functions for antiSMASH
Andrew Kiruluta, 05/22/2023
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import tqdm

# Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) 
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Position-wise Feed-Forward Networks
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Postional Encoding  
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attn = self.norm1(x)
        attn_output = self.self_attn(attn, attn, attn, mask)
        x = x + self.dropout(attn_output)
        ff = self.norm1(x)
        ff_output = self.feed_forward(ff)
        x = x + self.dropout(ff_output)
        return x
    
# Decoder Layer - No AntiSmash Encoder
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn1 = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output_1, src_mask_1, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output_1 = self.cross_attn1(x, enc_output_1, enc_output_1, src_mask_1)
        x = self.norm2(x + self.dropout(attn_output_1))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
# AntiSmash Arm Decoder Layer
class DecoderLayerAntismash(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayerAntismash, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn1 = MultiHeadAttention(d_model, num_heads)
        self.cross_attn2 = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
        
    def forward(self, x, enc_output_1, src_mask_1,enc_output_2, src_mask_2, tgt_mask):
        attnx = self.norm1(x)
        attn_output = self.self_attn(attnx, attnx, attnx, tgt_mask)
        x = x + self.dropout(attn_output)
        attn1  = self.norm1(x)
        attn_output_1 = self.cross_attn1(attn1, enc_output_1, enc_output_1, src_mask_1)
        x = x + self.dropout(attn_output_1)
        attn2 = self.norm1(x)
        attn_output_2 = self.cross_attn2(attn2, enc_output_2, enc_output_2, src_mask_2)
        x = x + self.dropout(attn_output_2)
        ff_in = self.norm1(x)
        ff_output = self.feed_forward(ff_in)
        x = x + self.dropout(ff_output)   
        return x
    
# use cross attention head to fuse the  only pfam encoder
class TransformerCA(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(TransformerCA, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_seq_length = max_seq_length

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        device = src.device
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(device)
        return src_mask, tgt_mask

    def forward(self, inp, inp2, tgt):
        src_mask, tgt_mask = self.generate_mask(inp, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(inp)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output,enc_output,src_mask,tgt_mask)

        output = self.fc(dec_output)
        return output
    
# use cross attention head to fuse the pfmam & antismash encoders
class TransformerCA_Antismash(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(TransformerCA_Antismash, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder1_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.encoder2_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        #self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.decoder_layers = nn.ModuleList([DecoderLayerAntismash(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_seq_length = max_seq_length

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        device = src.device
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(device)
        return src_mask, tgt_mask

    def forward(self, inp1, inp2, tgt):
        src_mask_1, tgt_mask = self.generate_mask(inp1, tgt)
        src_mask_2, _ = self.generate_mask(inp2, tgt)
        src_embedded_1 = self.dropout(self.positional_encoding(self.encoder_embedding(inp1)))
        src_embedded_2 = self.dropout(self.positional_encoding(self.encoder_embedding(inp2)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output_1 = src_embedded_1
        for enc_layer in self.encoder1_layers:
            enc_output_1 = enc_layer(enc_output_1, src_mask_1)

        enc_output_2 = src_embedded_2
        for enc_layer in self.encoder2_layers:
            enc_output_2 = enc_layer(enc_output_2, src_mask_2)
        
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output,enc_output_1,src_mask_1,enc_output_2,src_mask_2, tgt_mask)

        output = self.fc(dec_output)
        return output
    
