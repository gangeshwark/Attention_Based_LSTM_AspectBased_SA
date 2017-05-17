import numpy as np
import h5py

def get_emb(word, h):

    if word in h:
        return h[word][:]
    else:
        return h['__UNK__'][:]

with open('data/text_vocab.vocab', 'r') as f:
    h = h5py.File('data/text_vector.hdf5')
    lines = f.readlines()
    emb = []
    print(len(lines))
    for line in lines:
        i, word = line.strip().split('\t')
        emb.append(get_emb(word, h))
    print(len(emb), len(emb[0]))
    emb = np.asarray(emb)
    print(emb.shape)