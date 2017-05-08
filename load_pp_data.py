import pickle
from time import time

import operator
import pandas as pd
import gensim


def get_vocab():
    a = pd.read_pickle('restaurants_train_data_processed.pkl')
    b = pd.read_pickle('restaurants_test_data_processed.pkl')
    text_vocab = {}
    for x in a['text']:
        for word in x:
            if word in text_vocab.keys():
                text_vocab[word] += 1
            else:
                text_vocab[word] = 1
    for x in b['text']:
        for word in x:
            if word in text_vocab.keys():
                text_vocab[word] += 1
            else:
                text_vocab[word] = 1
    text_vocab = reversed(sorted(text_vocab.items(), key=operator.itemgetter(1)))
    aspect_vocab = {}
    for word in a['aspect']:
        if word == 'anecdotes/miscellaneous':
            word = 'miscellaneous'
        if word in aspect_vocab.keys():
            aspect_vocab[word] += 1
        else:
            aspect_vocab[word] = 1

    for word in b['aspect']:
        if word == 'anecdotes/miscellaneous':
            word = 'miscellaneous'
        if word in aspect_vocab.keys():
            aspect_vocab[word] += 1
        else:
            aspect_vocab[word] = 1

    aspect_vocab = reversed(sorted(aspect_vocab.items(), key=operator.itemgetter(1)))

    return list(text_vocab), list(aspect_vocab)


def get_vectors(text_vocab, aspect_vocab):
    text_skipped = 0
    aspect_skipped = 0
    st = time()
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '/home/gangeshwark/test_Google/GoogleNews-vectors-negative300.bin', binary=True)
    print time() - st, " seconds to load the Google News vectors."
    text_vector = {}
    for i, word in enumerate(text_vocab):
        if word[0] in text_vector.keys():
            continue
        try:
            text_vector[word[0]] = model[word[0]]
        except:
            text_skipped += 1

    aspect_vector = {}
    for i, word in enumerate(aspect_vocab):
        if word[0] in aspect_vector.keys():
            continue
        try:
            aspect_vector[word[0]] = model[word[0]]
        except:
            aspect_skipped += 1
    print "Skipped %d words from text and %d words from aspects"%(text_skipped, aspect_skipped)
    return text_vector, aspect_vector


if __name__ == '__main__':
    text_vocab, aspect_vocab = get_vocab()
    print text_vocab
    print len(text_vocab)
    print aspect_vocab
    print len(aspect_vocab)
    text_vector, aspect_vector = get_vectors(text_vocab, aspect_vocab)
    print len(text_vector), len(aspect_vector)
    with open('text_vector.pkl', 'wb') as f:
        pickle.dump(text_vector, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('aspect_vector.pkl', 'wb') as f:
        pickle.dump(aspect_vector, f, protocol=pickle.HIGHEST_PROTOCOL)
    print aspect_vector