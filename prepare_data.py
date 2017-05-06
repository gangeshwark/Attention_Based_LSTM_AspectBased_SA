import numpy as np
import pandas as pd
from tqdm import tqdm

import lxml
import xml.etree.ElementTree
from pprint import pprint


def get_laptop_data():
    laptop_df = pd.DataFrame(columns=('sentence_id', 'text', 'aspect', 'polarity', 'value_from', 'value_to'))

    e = xml.etree.ElementTree.parse('data/SemEval14-ABSA-TrainData_v2/Laptop_Train_v2.xml').getroot()

    pprint(e)
    sentences = e.findall('sentence')
    i = 0
    for sentence in tqdm(sentences):
        #print '\n'
        id = sentence.get('id')
        text = sentence.find('text').text
        #print text
        aspects = sentence.findall('aspectTerms')
        if len(aspects) == 0:
            a, p, f, t = None, None, None, None
            laptop_df.loc[i] = [id, text, a, p, f, t]
            i+=1
        else:
            for aspect in aspects[0].findall('aspectTerm'):
                a = aspect.get('term')
                p = aspect.get('polarity')
                f = aspect.get('from')
                t = aspect.get('to')
                laptop_df.loc[i] = [id, text, a, p, f, t]
                i+=1

    pprint(laptop_df)
    return laptop_df


def get_restaurants_train_data():
    restaurants_df = pd.DataFrame(
        columns=('sentence_id', 'text', 'aspect', 'polarity'))
    e = xml.etree.ElementTree.parse('data/SemEval14-ABSA-TrainData_v2/Restaurants_Train_v2.xml').getroot()

    pprint(e)
    sentences = e.findall('sentence')
    i = 0
    for sentence in tqdm(sentences):
        #print '\n'
        id = sentence.get('id')
        text = sentence.find('text').text
        #print text
        aspects = sentence.findall('aspectCategories')
        if len(aspects) == 0:
            a, p = None, None
            restaurants_df.loc[i] = [id, text, a, p]
            i+=1
        else:
            for aspect in aspects[0].findall('aspectCategory'):
                a = aspect.get('category')
                p = aspect.get('polarity')
                restaurants_df.loc[i] = [id, text, a, p]
                i+=1

    #pprint(restaurants_df)
    return restaurants_df

def get_restaurants_test_data():
    restaurants_df = pd.DataFrame(
        columns=('sentence_id', 'text', 'aspect', 'polarity'))
    e = xml.etree.ElementTree.parse('data/ABSA_TestData_PhaseB/Restaurants_Test_Data_phaseB.xml').getroot()
    #e = xml.etree.ElementTree.parse('data/SemEval14-ABSA-TrainData_v2/restaurants-trial.xml').getroot()

    pprint(e)
    sentences = e.findall('sentence')
    i = 0
    for sentence in tqdm(sentences):
        #print '\n'
        id = sentence.get('id')
        text = sentence.find('text').text
        #print text
        aspects = sentence.findall('aspectCategories')
        if len(aspects) == 0:
            a, p = None, None
            restaurants_df.loc[i] = [id, text, a, p]
            i+=1
        else:
            for aspect in aspects[0].findall('aspectCategory'):
                a = aspect.get('category')
                p = aspect.get('polarity')
                restaurants_df.loc[i] = [id, text, a, p]
                i+=1

    #pprint(restaurants_df)
    return restaurants_df


if __name__ == '__main__':
    #get_laptop_data()
    restaurants_train_data = get_restaurants_train_data()
    print restaurants_train_data.groupby('polarity').count()
    restaurants_train_data.to_csv('restaurants_train_data.tsv', '\t')

    restaurants_test_data = get_restaurants_test_data()
    print restaurants_test_data
    restaurants_test_data.to_csv('restaurants_test_data.tsv', "\t")