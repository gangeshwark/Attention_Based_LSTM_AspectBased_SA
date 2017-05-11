# convert a sentence into sentence of word ids
import ast

import pandas as pd
import numpy as np


def convert_sent_ids_with_pad(text, w2i, l, max_len):
    id = []
    for x in text:
        if x in w2i:
            id.append(w2i[x])
        else:
            id.append(w2i['__UNK__'])
    rem = max_len - l
    for x in xrange(rem):
        id.append(w2i['__PAD__'])

    assert max_len == len(id)
    # print id
    return id


def get_w2i():
    w2i = {}
    i2w = {}
    with open('../data/text_vocab.vocab', 'r') as f:
        lines = f.readlines()
        for x in lines:
            i, word = x.strip().split('\t')
            w2i[word] = i
            i2w[i] = word
    return w2i, i2w


def get_a2i():
    a2i = {}
    i2a = {}
    with open('../data/aspect_vocab.vocab', 'r') as f:
        lines = f.readlines()
        for x in lines:
            i, word = x.strip().split('\t')
            a2i[word] = i
            i2a[i] = word

    return a2i, i2a


def get_train_data():
    # load text vocab
    w2i, i2w = get_w2i()
    a2i, i2a = get_a2i()

    a = pd.read_hdf('../data/restaurants_train_data_processed.h5', 'table')
    a['text'] = a['text'].apply(ast.literal_eval)
    a = a[a.polarity != 'conflict']
    # shuffle dataset
    a = a.sample(frac=1).reset_index(drop=True)
    lens = []
    for t in a['text']:
        lens.append(len(t))
    max_len = 80
    a.loc[:, 'seq_len'] = pd.Series(lens, index=a.index)
    a.loc[:, 'max_len'] = pd.Series([max_len] * len(lens), index=a.index)
    # print a
    print a['text'][8]
    print a['polarity'][8]
    print a['aspect'][8]
    print a['seq_len'][8]
    print a['max_len'][8]
    for i, data in a.iterrows():

        a.set_value(i, 'text', convert_sent_ids_with_pad(data['text'], w2i, data['seq_len'], data['max_len']))
        asp = data['aspect']
        if asp == 'anecdotes/miscellaneous':
            asp = 'miscellaneous'
        a.set_value(i, 'aspect', a2i[asp])
        p = data['polarity']
        if p == 'negative':
            a.set_value(i, 'polarity', [1, 0, 0])
        elif p == 'positive':
            a.set_value(i, 'polarity', [0, 0, 1])
        elif p == 'neutral':
            a.set_value(i, 'polarity', [0, 1, 0])

    print a
    print "\n\n", a['text'][8]
    print a['polarity'][8]
    print a['aspect'][8]
    a.to_hdf('../data/train_data_3classes.h5', 'table')


def get_test_data():
    # load text vocab
    w2i = {}
    i2w = {}
    with open('../data/text_vocab.vocab', 'r') as f:
        lines = f.readlines()
        for x in lines:
            i, word = x.strip().split('\t')
            w2i[word] = i
            i2w[i] = word
    print w2i
    print len(w2i)
    a2i = {}
    i2a = {}
    with open('../data/aspect_vocab.vocab', 'r') as f:
        lines = f.readlines()
        for x in lines:
            i, word = x.strip().split('\t')
            a2i[word] = i
            i2a[i] = word
    print a2i
    print len(a2i)

    a = pd.read_hdf('../data/restaurants_test_data_processed.h5', 'table')
    a['text'] = a['text'].apply(ast.literal_eval)
    # shuffle dataset
    a = a.sample(frac=1).reset_index(drop=True)
    lens = []
    for t in a['text']:
        lens.append(len(t))
    max_len = 80
    a.loc[:, 'seq_len'] = pd.Series(lens, index=a.index)
    a.loc[:, 'max_len'] = pd.Series([max_len] * len(lens), index=a.index)
    # print a
    print a['text'][8]
    print a['aspect'][8]
    print a['seq_len'][8]
    print a['max_len'][8]
    for i, data in a.iterrows():
        a.set_value(i, 'text', convert_sent_ids_with_pad(data['text'], w2i, data['seq_len'], data['max_len']))
        asp = data['aspect']
        if asp == 'anecdotes/miscellaneous':
            asp = 'miscellaneous'
        a.set_value(i, 'aspect', a2i[asp])

    print a
    print "\n\n", a['text'][8]
    print a['aspect'][8]
    a.to_hdf('../data/test_data_3classes.h5', 'table')


if __name__ == '__main__':
    get_train_data()
    get_test_data()
