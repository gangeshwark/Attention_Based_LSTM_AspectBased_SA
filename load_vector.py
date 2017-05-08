import pandas as pd

a = pd.read_pickle('text_vector.pkl')
b = pd.read_pickle('aspect_vector.pkl')
print len(a), len(b)
print b['food']