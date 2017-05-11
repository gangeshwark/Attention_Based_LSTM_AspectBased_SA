import pandas as pd
import numpy as np


class TrainData():
    def __init__(self, batch_size, input_len):
        # load training data
        self.a = pd.read_hdf('../data/train_data_3classes.h5', 'table')
        self.bz = batch_size
        self.i = 0
        self.len = input_len

    def __iter__(self):
        return self

    def next(self):
        x = self.a['text'][self.i:self.i + self.bz]
        a = self.a['aspect'][self.i:self.i + self.bz]
        y = self.a['polarity'][self.i:self.i + self.bz]
        x_len = self.a['seq_len'][self.i:self.i + self.bz]
        # print x
        self.i += self.bz

        x_ = []
        a_ = []
        y_ = []
        for i in x:
            x_.append(np.asarray(map(int, i)))
        for i in y:
            y_.append(np.asarray(map(int, i)))
        for i in a:
            a_.append(int(i))
        x = np.asarray(x_)
        a = np.asarray(a_)
        y = np.asarray(y_, dtype=np.int64)
        x_len = np.asarray(x_len, dtype=np.int64)
        return x, x_len, a, y


class EvalData():
    pass


class TestData():
    def __init__(self, batch_size, input_len):
        # load training data
        self.a = pd.read_hdf('../data/test_data_3classes.h5', 'table')
        self.bz = batch_size
        self.i = 0
        self.len = input_len

    def __iter__(self):
        return self

    def next(self):
        x = self.a['text'][self.i:self.i + self.bz]
        a = self.a['aspect'][self.i:self.i + self.bz]
        x_len = self.a['seq_len'][self.i:self.i + self.bz]
        self.i += self.bz

        x_ = []
        a_ = []
        for i in x:
            x_.append(np.asarray(map(int, i)))

        for i in a:
            a_.append(int(i))
        x = np.asarray(x_)
        a = np.asarray(a_)
        x_len = np.asarray(x_len, dtype=np.int64)


        return x,x_len, a



# testing
if __name__ == '__main__':
    data = TrainData(32)
    i = 0
    while (1):
        i += 1
        x, a, y = next(data)
        print a
        print len(x)
        if len(x) < 1:
            break
        print '______________________________________________________________________'

    print "i", i

    data = TestData(32)
    i = 0
    while (1):
        i += 1
        x, a = next(data)
        print len(x)
        if len(x) < 1:
            break
        print '______________________________________________________________________'

    print "i", i
