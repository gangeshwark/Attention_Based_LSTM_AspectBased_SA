import numpy as np
import h5py
import pickle

def get_emb(word, h):

    if word in h:
        return h[word][:]
    else:
        return h['__UNK__'][:]

path = 'data/semeval16/laptop/text_vocab.vocab'

with open(path, 'r') as f:
    with open('data/semeval16/laptop/text_vector.pkl', 'rb') as pkl_file:
        h = pickle.load(pkl_file)
        print(type(h))
        lines = f.readlines()
        emb = []
        print(len(lines))
        for line in lines:
            i, word = line.strip().split('\t')
            emb.append(get_emb(word, h))
        print(len(emb), len(emb[0]))
        emb = np.asarray(emb)
        print(emb.shape)