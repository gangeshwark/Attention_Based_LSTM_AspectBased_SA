# this file contains logic to create data that is required for training

import pandas as pd


# convert a sentence into sentence of word ids
def convert_sent_ids_with_pad(text, w2i, l, max_len):
    id = []
    for x in text:
        if x in w2i:
            id.append(w2i[x])
        else:
            id.append(w2i['__UNK__'])
    rem = max_len - l
    for x in range(rem):
        id.append(w2i['__PAD__'])

    assert max_len == len(id)
    # print id
    return id


def get_w2i():
    w2i = {}
    i2w = {}
    with open('../data/semeval14/text_vocab.vocab', 'r') as f:
        lines = f.readlines()
        for x in lines:
            i, word = x.strip().split('\t')
            w2i[word] = i
            i2w[i] = word
    return w2i, i2w


def get_a2i():
    a2i = {}
    i2a = {}
    with open('../data/semeval14/aspect_vocab.vocab', 'r') as f:
        lines = f.readlines()
        for x in lines:
            i, word = x.strip().split('\t')
            a2i[word] = i
            i2a[i] = word

    return a2i, i2a


def create_train_data(a, i2w, i2a, w2i, a2i, save_path):
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
    print(a['text'][8])
    print(a['polarity'][8])
    print(a['aspect'][8])
    print(a['seq_len'][8])
    print(a['max_len'][8])
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

    print(a)
    print("\n\n", a['text'][8])
    print(a['polarity'][8])
    print(a['aspect'][8])
    a.to_csv(save_path + '/rest_train_data.tsv', "\t")
    a.to_pickle(save_path + '/rest_train_data.pkl')


def create_test_data(a, i2w, i2a, w2i, a2i, save_path):
    # shuffle dataset
    a = a.sample(frac=1).reset_index(drop=True)
    lens = []
    for t in a['text']:
        lens.append(len(t))
    max_len = 80
    a.loc[:, 'seq_len'] = pd.Series(lens, index=a.index)
    a.loc[:, 'max_len'] = pd.Series([max_len] * len(lens), index=a.index)
    # print a
    print(a['text'][8])
    print(a['aspect'][8])
    print(a['seq_len'][8])
    print(a['max_len'][8])
    for i, data in a.iterrows():
        a.set_value(i, 'text', convert_sent_ids_with_pad(data['text'], w2i, data['seq_len'], data['max_len']))
        asp = data['aspect']
        if asp == 'anecdotes/miscellaneous':
            asp = 'miscellaneous'
        a.set_value(i, 'aspect', a2i[asp])

    print(a)
    print("\n\n", a['text'][8])
    print(a['aspect'][8])
    a.to_csv(save_path + '/rest_test_data.tsv', "\t")
    a.to_pickle(save_path + '/rest_test_data.pkl')
