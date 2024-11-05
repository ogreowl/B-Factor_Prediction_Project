# src/feature_engineering.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from transformers import BertModel, BertTokenizer
import torch

def prepare_features_with_position(sequence: str) -> np.ndarray:
    """Creates one-hot encoding and normalized positional encoding for sequence."""
    amino_acids_df = pd.DataFrame(list(sequence), columns=['amino_acid'])
    encoder = OneHotEncoder(sparse_output=False)
    X_one_hot = encoder.fit_transform(amino_acids_df)
    positions = np.arange(len(sequence)) / len(sequence)
    return np.hstack((X_one_hot, positions.reshape(-1, 1)))

def generate_probert_embeddings(sequence: str) -> np.ndarray:
    """Generates ProBERT embeddings for a given amino acid sequence."""
    model = BertModel.from_pretrained("Rostlab/prot_bert", output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
    sequence = " ".join(sequence)
    tokens = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    residue_embeddings = outputs.hidden_states[-1].squeeze(0)[1:-1]
    return residue_embeddings.cpu().numpy()
