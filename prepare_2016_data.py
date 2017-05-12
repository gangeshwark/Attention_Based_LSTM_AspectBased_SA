import numpy as np
import pandas as pd
from tqdm import tqdm
import lxml
import xml.etree.ElementTree
from pprint import pprint

def get_data(path):
    print "Preparing data.."
    e = xml.etree.ElementTree.parse(path).getroot()

    restaurants_df = pd.DataFrame(
        columns=('review_id', 'sentence_id', 'text', 'target', 'category', 'polarity'))

    # pprint(e)
    Reviews = e.findall('Review')
    # print Reviews
    i = 0
    for review in tqdm(Reviews):
        # print '\n'
        r_id = review.get('rid')
        sentences = review.findall('sentences')
        # print sentences
        for sentence in sentences[0]:
            # print sentence
            id = sentence.get('id')
            text = sentence.find('text').text
            # print text
            Opinions = sentence.findall('Opinions')
            if len(Opinions) == 0:
                t, a, p = None, None, None
                restaurants_df.loc[i] = [r_id, id, text, t, a, p]
                i += 1
            else:
                for Opinion in Opinions[0].findall('Opinion'):
                    t = Opinion.get('target')
                    a = Opinion.get('category')
                    p = Opinion.get('polarity')
                    restaurants_df.loc[i] = [r_id, id, text, t, a, p]
                    i += 1

    # pprint(restaurants_df)
    return restaurants_df


if __name__ == '__main__':
    # get_laptop_data()
    restaurants_train_data = get_data('data/raw_data/SemEval_16/ABSA16_Restaurants_Train_SB1_v2.xml')
    #print restaurants_train_data.groupby('polarity').count()
    print restaurants_train_data.shape[0], " data points"
    restaurants_train_data.to_csv('data/semeval16/restaurants_train_data.tsv', '\t', encoding='utf-8')

    restaurants_test_data = get_data('data/raw_data/SemEval_16/EN_REST_SB1_TEST.xml.gold')
    #print restaurants_train_data.groupby('polarity').count()
    print restaurants_test_data.shape[0], " data points"
    restaurants_test_data.to_csv('data/semeval16/restaurants_test_data.tsv', '\t', encoding='utf-8')

    laptop_train_data = get_data('data/raw_data/SemEval_16/ABSA16_Laptops_Train_SB1_v2.xml')
    # print restaurants_train_data.groupby('polarity').count()
    print laptop_train_data.shape[0], " data points"
    laptop_train_data.to_csv('data/semeval16/laptop_train_data.tsv', '\t', encoding='utf-8')

    laptop_test_data = get_data('data/raw_data/SemEval_16/EN_LAPT_SB1_TEST_.xml.gold')
    # print restaurants_train_data.groupby('polarity').count()
    print laptop_test_data.shape[0], " data points"
    laptop_test_data.to_csv('data/semeval16/laptop_test_data.tsv', '\t', encoding='utf-8')

    train_data = restaurants_train_data.append(laptop_train_data, ignore_index=True)
    train_data.to_csv('data/semeval16/train_data.tsv', '\t', encoding='utf-8')

    test_data = restaurants_test_data.append(laptop_test_data, ignore_index=True)
    test_data.to_csv('data/semeval16/test_data.tsv', '\t', encoding='utf-8')


    # restaurants_test_data = get_restaurants_test_data()
    # print restaurants_test_data
    # restaurants_test_data.to_csv('data/restaurants_test_data.tsv', "\t")
