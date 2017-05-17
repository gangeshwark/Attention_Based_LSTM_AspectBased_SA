import ast
import pickle
from time import time

import operator

import h5py
import pandas as pd
import gensim
import numpy as np


def get_vocab(a, b):
    text_vocab = {}
    for x in a['text']:
        for word in x:
            if word in list(text_vocab.keys()):
                text_vocab[word] += 1
            else:
                text_vocab[word] = 1
    for x in b['text']:
        for word in x:
            if word in list(text_vocab.keys()):
                text_vocab[word] += 1
            else:
                text_vocab[word] = 1
    text_vocab = reversed(sorted(list(text_vocab.items()), key=operator.itemgetter(1)))
    entity_vocab = {}
    for word in a['entity']:
        if not word:
            continue
        if word.lower() in list(entity_vocab.keys()):
            entity_vocab[word.lower()] += 1
        else:
            entity_vocab[word.lower()] = 1

    for word in b['entity']:
        if not word:
            continue
        if word.lower() in list(entity_vocab.keys()):
            entity_vocab[word.lower()] += 1
        else:
            entity_vocab[word.lower()] = 1

    entity_vocab = reversed(sorted(list(entity_vocab.items()), key=operator.itemgetter(1)))

    attribute_vocab = {}
    for word in a['attribute']:
        if not word:
            continue
        if word == 'STYLE_OPTIONS':
            word = 'STYLE'
        if word.lower() in list(attribute_vocab.keys()):
            attribute_vocab[word.lower()] += 1
        else:
            attribute_vocab[word.lower()] = 1

    for word in b['attribute']:
        if not word:
            continue
        if word == 'STYLE_OPTIONS':
            word = 'STYLE'
        if word.lower() in list(attribute_vocab.keys()):
            attribute_vocab[word.lower()] += 1
        else:
            attribute_vocab[word.lower()] = 1

    attribute_vocab = reversed(sorted(list(attribute_vocab.items()), key=operator.itemgetter(1)))

    return list(text_vocab), list(entity_vocab), list(attribute_vocab)


def get_vectors(text_vocab, entity_vocab, attribute_vocab):
    text_skipped = 0
    entity_skipped = 0
    attribute_skipped = 0
    st = time()
    # Load Google's pre-trained Word2Vec model.
    print('Loading Google News Word2Vec model')
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '/home/gangeshwark/test_Google/GoogleNews-vectors-negative300.bin', binary=True)
    print(time() - st, " seconds to load the Google News vectors.")

    unk = np.random.uniform(-np.sqrt(3.0), np.sqrt(3.0), 300)
    pad = np.random.uniform(-np.sqrt(3.0), np.sqrt(3.0), 300)
    period = np.random.uniform(-np.sqrt(3.0), np.sqrt(3.0), 300)
    text_vector = {'__UNK__': unk, '__PAD__': pad, '.': period}
    for i, word in enumerate(text_vocab):
        if word[0] in list(text_vector.keys()):
            continue
        try:
            text_vector[word[0]] = model[word[0]]
        except:
            text_skipped += 1

    entity_vector = {'__UNK__': unk}
    for i, word in enumerate(entity_vocab):
        if word[0] in list(entity_vector.keys()):
            continue
        try:
            entity_vector[word[0]] = model[word[0]]
        except:
            entity_skipped += 1

    attribute_vector = {'__UNK__': unk}
    for i, word in enumerate(attribute_vocab):

        if word[0] in list(attribute_vector.keys()):
            continue
        try:
            attribute_vector[word[0]] = model[word[0]]
        except:
            attribute_skipped += 1

    print("Skipped %d words from text and %d, %d words from entity and attribute" % (
        text_skipped, entity_skipped, attribute_skipped))
    return text_vector, entity_vector, attribute_vector


if __name__ == '__main__':
    raw_2014_path = '../../data/raw_data/SemEval_14'
    raw_2016_path = '../../data/raw_data/SemEval_14'
    p_2014_path = '../../data/semeval14'
    p_2016_path = '../../data/semeval16'

    text_vocab, entity_vocab, attribute_vocab = get_vocab()

    print(text_vocab)
    print(len(text_vocab))
    # contains all the words
    with open(p_2016_path + '/all_text_vocab.vocab', 'w') as f:
        for i, word in enumerate(sorted(text_vocab)):
            f.write('%d\t%s\n' % (i, word[0]))

    print(entity_vocab)
    print(len(entity_vocab))
    with open(p_2016_path + '/all_entity_vocab.vocab', 'w') as f:
        for i, word in enumerate(entity_vocab):
            f.write('%d\t%s\n' % (i, word[0]))

    print(attribute_vocab)
    print(len(attribute_vocab))
    with open(p_2016_path + '/all_attribute_vocab.vocab', 'w') as f:
        for i, word in enumerate(attribute_vocab):
            f.write('%d\t%s\n' % (i, word[0]))

    text_vector, entity_vector, attribute_vector = get_vectors(text_vocab, entity_vocab, attribute_vocab)

    # contains only the words that have embeddings
    with open(p_2016_path + '/text_vocab.vocab', 'w') as f:
        for i, word in enumerate(sorted(list(text_vector.keys()))):
            f.write('%d\t%s\n' % (i, word))

    with open(p_2016_path + '/entity_vocab.vocab', 'w') as f:
        for i, word in enumerate(sorted(list(entity_vector.keys()))):
            f.write('%d\t%s\n' % (i, word))

    with open(p_2016_path + '/attribute_vocab.vocab', 'w') as f:
        for i, word in enumerate(sorted(list(attribute_vector.keys()))):
            f.write('%d\t%s\n' % (i, word))

    text_dict = dict(enumerate(sorted(list(text_vector.keys()))))
    entity_dict = dict(enumerate(sorted(list(entity_vector.keys()))))
    attribute_dict = dict(enumerate(sorted(list(attribute_vector.keys()))))

    with open(p_2014_path + '/text_vocab.pkl', 'wb') as f:
        pickle.dump(text_dict, f)
    with open(p_2014_path + '/entity_vocab.pkl', 'wb') as f:
        pickle.dump(entity_dict, f)
    with open(p_2014_path + '/attribute_vocab.pkl', 'wb') as f:
        pickle.dump(attribute_dict, f)

    print(len(text_vector), len(entity_vector), len(attribute_vector))

    with open(p_2016_path + '/text_vector.pkl', 'wb') as f:
        pickle.dump(text_vector, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(p_2016_path + '/entity_vector.pkl', 'wb') as f:
        pickle.dump(entity_vector, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(p_2016_path + '/attribute_vector.pkl', 'wb') as f:
        pickle.dump(attribute_vector, f, protocol=pickle.HIGHEST_PROTOCOL)
