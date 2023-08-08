This repository contains code for deep learning based annotation and prediction of BGC structures from BGC protein family sequences using an NLP transformer architecture. The files are annotated as follows:

- trainer_torch_antismash.py: Main wrapper script for the model. To run the model:
     $ python trainer_torch_antismash.py
     see the argparse module in the script for configurations

- utils_antismash.py: contains tokenizers for smiles, selfies and BGC protein families.

- train_tokenizer.py: custom trainer for protein family tokenizer to create alphabet for pfam (byte-level-bpe.tokenizer.json). It creates the alphabet that is used by utils_antismash.py

- dataset_antismash.py: Pytorch dataset class for reading in training data from disk.

- pretrainer_antismash.py: script for pre-training of the transformer decoder with the a sampled ZINC dataset. In this implementation, the cross-attention head in the decoder is frozen while the self-attention head and the fully connected layers are  trainable. Random masking of input tokens similar to that of the BERT implementation is used during training.

- Transformer_torch_antismash.ipynb: notebook prototype for the BGC transformer model.

