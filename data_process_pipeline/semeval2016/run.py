import pickle

from data_process_pipeline.semeval2016.load_pp_data import get_vocab, get_vectors
from data_process_pipeline.semeval2016.prepare_2016_data import get_data
from data_process_pipeline.semeval2016.preprocess import clean


def prepare_data(folder):
    raw_2016_path = '../../data/raw_data/SemEval_16'
    p_2016_path = '../../data/semeval16/' + folder
    # get_laptop_data()

    if folder == 'restaurants':
        print('Yes, rest')
        train_data = get_data(raw_2016_path + '/ABSA16_Restaurants_Train_SB1_v2.xml')
        test_data = get_data(raw_2016_path + '/EN_REST_SB1_TEST.gold.xml')
    elif folder == 'laptop':
        print('Yes, laptop')
        train_data = get_data(raw_2016_path + '/ABSA16_Laptops_Train_SB1_v2.xml')
        test_data = get_data(raw_2016_path + '/EN_LAPT_SB1_TEST_.gold.xml')
    else:
        return

    print(train_data.shape[0], " data points")
    train_data.to_csv(p_2016_path + '/train_data.tsv', '\t', encoding='utf-8')

    print(test_data.shape[0], " data points")
    test_data.to_csv(p_2016_path + '/test_data.tsv', '\t', encoding='utf-8')

    train_data['text'] = train_data['text'].apply(clean)
    test_data['text'] = test_data['text'].apply(clean)

    # save pre-processed data as pickle file
    train_data.to_pickle(p_2016_path + '/train_data_processed.pkl')
    test_data.to_pickle(p_2016_path + '/test_data_processed.pkl')
    #print(test_data)

    def get_vec(text_vocab, entity_vocab, attribute_vocab):
        print(text_vocab)
        print(len(text_vocab))
        # contains all the words
        with open(p_2016_path + '/all_text_vocab.vocab', 'w') as f:
            for i, word in enumerate(sorted(text_vocab)):
                f.write('%d\t%s\n' % (i, word[0]))

        print(entity_vocab)
        print(len(entity_vocab))
        with open(p_2016_path + '/all_entity_vocab.vocab', 'w') as f:
            for i, word in enumerate(sorted(entity_vocab)):
                f.write('%d\t%s\n' % (i, word[0]))

        print(attribute_vocab)
        print(len(attribute_vocab))
        with open(p_2016_path + '/all_attribute_vocab.vocab', 'w') as f:
            for i, word in enumerate(sorted(attribute_vocab)):
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

        with open(p_2016_path + '/text_vocab.pkl', 'wb') as f:
            pickle.dump(text_dict, f)
        with open(p_2016_path + '/entity_vocab.pkl', 'wb') as f:
            pickle.dump(entity_dict, f)
        with open(p_2016_path + '/attribute_vocab.pkl', 'wb') as f:
            pickle.dump(attribute_dict, f)

        print(len(text_vector), len(entity_vector), len(attribute_vector))

        with open(p_2016_path + '/text_vector.pkl', 'wb') as f:
            pickle.dump(text_vector, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(p_2016_path + '/entity_vector.pkl', 'wb') as f:
            pickle.dump(entity_vector, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(p_2016_path + '/attribute_vector.pkl', 'wb') as f:
            pickle.dump(attribute_vector, f, protocol=pickle.HIGHEST_PROTOCOL)

    text_vocab, entity_vocab, attribute_vocab = get_vocab(train_data, test_data)
    get_vec(text_vocab, entity_vocab, attribute_vocab)


if __name__ == '__main__':
    prepare_data('restaurants')
    prepare_data('laptop')

    """
    # get_laptop_data()
    restaurants_train_data = get_data(raw_2016_path + '/ABSA16_Restaurants_Train_SB1_v2.xml')
    # print restaurants_train_data.groupby('polarity').count()
    print(restaurants_train_data.shape[0], " data points")
    restaurants_train_data.to_csv(p_2016_path + '/restaurants_train_data.tsv', '\t', encoding='utf-8')

    restaurants_test_data = get_data(raw_2016_path + '/EN_REST_SB1_TEST.gold.xml')
    # print restaurants_train_data.groupby('polarity').count()
    print(restaurants_test_data.shape[0], " data points")
    restaurants_test_data.to_csv(p_2016_path + '/restaurants_test_data.tsv', '\t', encoding='utf-8')

    laptop_train_data = get_data(raw_2016_path + '/ABSA16_Laptops_Train_SB1_v2.xml')
    # print restaurants_train_data.groupby('polarity').count()
    print(laptop_train_data.shape[0], " data points")
    laptop_train_data.to_csv(p_2016_path + '/laptop_train_data.tsv', '\t', encoding='utf-8')

    laptop_test_data = get_data(raw_2016_path + '/EN_LAPT_SB1_TEST_.gold.xml')
    # print restaurants_train_data.groupby('polarity').count()
    print(laptop_test_data.shape[0], " data points")
    laptop_test_data.to_csv(p_2016_path + '/laptop_test_data.tsv', '\t', encoding='utf-8')

    train_data = restaurants_train_data.append(laptop_train_data, ignore_index=True)
    train_data.to_csv(p_2016_path + '/train_data.tsv', '\t', encoding='utf-8')

    test_data = restaurants_test_data.append(laptop_test_data, ignore_index=True)
    test_data.to_csv(p_2016_path + '/test_data.tsv', '\t', encoding='utf-8')
    # data = train_data.append(test_data, ignore_index=True)

    restaurants_train_data['text'] = restaurants_train_data['text'].apply(clean)
    restaurants_test_data['text'] = restaurants_test_data['text'].apply(clean)

    # save pre-processed data as pickle file
    restaurants_train_data.to_pickle(p_2016_path + '/restaurants_train_data_processed.pkl')
    restaurants_test_data.to_pickle(p_2016_path + '/restaurants_test_data_processed.pkl')
    print(restaurants_test_data)

    def get_vec(text_vocab, entity_vocab, attribute_vocab):
        print(text_vocab)
        print(len(text_vocab))
        # contains all the words
        with open(p_2016_path + '/all_text_vocab.vocab', 'w') as f:
            for i, word in enumerate(sorted(text_vocab)):
                f.write('%d\t%s\n' % (i, word[0]))

        print(entity_vocab)
        print(len(entity_vocab))
        with open(p_2016_path + '/all_entity_vocab.vocab', 'w') as f:
            for i, word in enumerate(sorted(entity_vocab)):
                f.write('%d\t%s\n' % (i, word[0]))

        print(attribute_vocab)
        print(len(attribute_vocab))
        with open(p_2016_path + '/all_attribute_vocab.vocab', 'w') as f:
            for i, word in enumerate(sorted(attribute_vocab)):
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

        with open(p_2016_path + '/text_vocab.pkl', 'wb') as f:
            pickle.dump(text_dict, f)
        with open(p_2016_path + '/entity_vocab.pkl', 'wb') as f:
            pickle.dump(entity_dict, f)
        with open(p_2016_path + '/attribute_vocab.pkl', 'wb') as f:
            pickle.dump(attribute_dict, f)

        print(len(text_vector), len(entity_vector), len(attribute_vector))

        with open(p_2016_path + '/text_vector.pkl', 'wb') as f:
            pickle.dump(text_vector, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(p_2016_path + '/entity_vector.pkl', 'wb') as f:
            pickle.dump(entity_vector, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(p_2016_path + '/attribute_vector.pkl', 'wb') as f:
            pickle.dump(attribute_vector, f, protocol=pickle.HIGHEST_PROTOCOL)


    text_vocab, entity_vocab, attribute_vocab = get_vocab(restaurants_train_data, restaurants_test_data)
    get_vec(text_vocab, entity_vocab, attribute_vocab)
    """
