import h5py
import pandas as pd


# a = pd.read_pickle('text_vector.pkl')
# b = pd.read_pickle('aspect_vector.pkl')
# print len(a), len(b)
# print b['food']

def get_word_vector_hdf5(hdf5_file, word):
    if hdf5_file[word]:
        return hdf5_file[word][:]
    else:
        return None


def get_all_word_vectors_hdf5(hdf5_file):
    wv = {}
    for x in hdf5_file:
        wv[x] = hdf5_file[x][:]

    return wv


# totally 4665 words vectors are available
if __name__ == '__main__':
    h = h5py.File('text_vector.hdf5')
    print get_word_vector_hdf5(h, 'hello')
    word_vectors = get_all_word_vectors_hdf5(h)
    print len(word_vectors)
