import ast
import pickle

import h5py
import pandas as pd

from data_process_pipeline.semeval2014.load_pp_data import get_vocab, get_vectors
from data_process_pipeline.semeval2014.prepare_2014_data import get_restaurants_train_data, get_restaurants_test_data
from data_process_pipeline.semeval2014.preprocess import clean

raw_2014_path = '../../data/raw_data/SemEval_14'
p_2014_path = '../../data/semeval14'

if __name__ == '__main__':
    # prepare data
    restaurants_train_data = get_restaurants_train_data(
        raw_2014_path + '/SemEval14-ABSA-TrainData_v2/Restaurants_Train_v2.xml')
    print(restaurants_train_data.groupby('polarity').count())
    restaurants_train_data.to_csv(p_2014_path + '/restaurants_train_data.tsv', '\t')

    restaurants_test_data = get_restaurants_test_data(
        raw_2014_path + '/ABSA_TestData_PhaseB/Restaurants_Test_Data_phaseB.xml')

    restaurants_test_data.to_csv(p_2014_path + '/restaurants_test_data.tsv', "\t")

    restaurants_train_data['text'] = restaurants_train_data['text'].apply(clean)
    restaurants_test_data['text'] = restaurants_test_data['text'].apply(clean)

    # save pre-processed data as pickle file
    restaurants_train_data.to_pickle(p_2014_path + '/restaurants_train_data_processed.pkl')
    restaurants_test_data.to_pickle(p_2014_path + '/restaurants_test_data_processed.pkl')
    print(restaurants_test_data)

    # load vocab and get vectors
    text_vocab, aspect_vocab = get_vocab(restaurants_train_data, restaurants_test_data)
    print(text_vocab)
    print(len(text_vocab))
    for i, word in enumerate(sorted(text_vocab)):
        print(i, word)

    def get_vec(text_vocab, aspect_vocab):

        # contains all the words
        with open(p_2014_path + '/all_text_vocab.vocab', 'w') as f:
            for i, word in enumerate(sorted(text_vocab)):
                f.write('%d\t%s\n' % (i, word[0]))

        print(aspect_vocab)
        print(len(aspect_vocab))
        with open(p_2014_path + '/all_aspect_vocab.vocab', 'w') as f:
            for i, word in enumerate(sorted(aspect_vocab)):
                f.write('%d\t%s\n' % (i, word[0]))

        text_vector, aspect_vector = get_vectors(text_vocab, aspect_vocab)

        # contains only the words that have embeddings
        with open(p_2014_path + '/text_vocab.vocab', 'w') as f:
            for i, word in enumerate(sorted(list(text_vector.keys()))):
                f.write('%d\t%s\n' % (i, word))

        with open(p_2014_path + '/aspect_vocab.vocab', 'w') as f:
            for i, word in enumerate(sorted(list(aspect_vector.keys()))):
                f.write('%d\t%s\n' % (i, word))

        text_dict = dict(enumerate(sorted(list(text_vector.keys()))))
        aspect_dict = dict(enumerate(sorted(list(aspect_vector.keys()))))
        with open(p_2014_path + '/text_vocab.pkl', 'wb') as f:
            pickle.dump(text_dict, f)
        with open(p_2014_path + '/aspect_vocab.pkl', 'wb') as f:
            pickle.dump(aspect_dict, f)

        print(len(text_vector), len(aspect_vector))
        with open(p_2014_path + '/text_vector.pkl', 'wb') as f:
            pickle.dump(text_vector, f)
        with open(p_2014_path + '/aspect_vector.pkl', 'wb') as f:
            pickle.dump(aspect_vector, f)

    get_vec(text_vocab, aspect_vocab)
