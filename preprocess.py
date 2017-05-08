import pandas as pd
import re
import string
from nltk import tokenize
a = pd.read_csv('restaurants_train_data.tsv', delimiter='\t')
b = pd.read_csv('restaurants_test_data.tsv', delimiter='\t')

def clean(s):
    s = re.sub('([' + string.punctuation + '])', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    tokenizer = tokenize.WhitespaceTokenizer()
    s = tokenizer.tokenize(s)
    #s = s.lower().split()
    return s

a['text'] = a['text'].apply(clean)
b['text'] = b['text'].apply(clean)

# save pre-processed data as pickle file
a.to_pickle('restaurants_train_data_processed.pkl')
b.to_pickle('restaurants_test_data_processed.pkl')
# load pre-processed pickle data
a = pd.read_pickle('restaurants_train_data_processed.pkl')
b = pd.read_pickle('restaurants_test_data_processed.pkl')
