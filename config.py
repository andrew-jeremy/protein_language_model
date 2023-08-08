max_seq_length = 300  # df1.pfam.str.len().quantile(0.9) = 263, df2.antismash.str.len().quantile(0.9) = 130
src_vocab_size = 3800 #  vocab size - 3655 unique pfam 
tgt_vocab_size = 52   # smile vocab
d_model = 512
num_heads = 16 
num_layers = 6
d_ff = 2048 
