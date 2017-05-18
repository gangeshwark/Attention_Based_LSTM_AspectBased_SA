import pandas as pd
import numpy as np


class TrainData():
    def __init__(self, batch_size, input_len):
        # load training data
        self.df = pd.read_pickle('../data/semeval14/rest_train_data.pkl')
        self.bz = batch_size
        self.i = 0
        self.len = input_len
        l = self.df.shape[0] - self.bz
        self.df = self.df.head(l)

    def __iter__(self):
        return self

    def __next__(self):
        x = self.df['text'][self.i:self.i + self.bz]
        a = self.df['aspect'][self.i:self.i + self.bz]
        y = self.df['polarity'][self.i:self.i + self.bz]
        x_len = self.df['seq_len'][self.i:self.i + self.bz]
        # print x
        self.i += self.bz

        x_ = []
        a_ = []
        y_ = []
        for i in x:
            x_.append(np.asarray(list(map(int, i))))
        for i in y:
            y_.append(np.asarray(list(map(int, i))))
        for i in a:
            a_.append(int(i))
        x = np.asarray(x_)
        a = np.asarray(a_)
        y = np.asarray(y_, dtype=np.int64)
        x_len = np.asarray(x_len, dtype=np.int64)
        return x, x_len, a, y


class EvalData():
    def __init__(self, batch_size, input_len):
        # load training data
        self.a = pd.read_pickle('../data/semeval14/rest_train_data.pkl')

        self.bz = batch_size
        self.i = 0
        self.len = input_len

    def __iter__(self):
        return self

    def __next__(self):
        x = self.a['text'].tail(self.bz)
        a = self.a['aspect'].tail(self.bz)

        y = self.a['polarity'].tail(self.bz)
        x_len = self.a['seq_len'].tail(self.bz)
        # self.i += self.bz

        x_ = []
        a_ = []
        y_ = []
        for i in x:
            x_.append(np.asarray(list(map(int, i))))
        for i in y:
            y_.append(np.asarray(list(map(int, i))))
        for i in a:
            a_.append(int(i))
        x = np.asarray(x_)
        a = np.asarray(a_)
        x_len = np.asarray(x_len, dtype=np.int64)

        return x, x_len, a, y


class TestData():
    def __init__(self, batch_size, input_len):
        # load training data
        self.a = pd.read_pickle('../data/semeval14/rest_train_data.pkl')
        self.bz = batch_size
        self.i = 0
        self.len = input_len

    def __iter__(self):
        return self

    def __next__(self):
        x = self.a['text'][self.i:self.i + self.bz]
        a = self.a['aspect'][self.i:self.i + self.bz]
        x_len = self.a['seq_len'][self.i:self.i + self.bz]
        # self.i += self.bz

        x_ = []
        a_ = []
        for i in x:
            x_.append(np.asarray(list(map(int, i))))

        for i in a:
            a_.append(int(i))
        x = np.asarray(x_)
        a = np.asarray(a_)
        x_len = np.asarray(x_len, dtype=np.int64)

        return x, x_len, a


# testing
if __name__ == '__main__':
    data = TrainData(25, 80)
    i = 0
    while (1):
        i += 1
        x, x_len, a, y = next(data)
        print(a, y)
        print(len(x))
        if len(x) < 1:
            break
        print('______________________________________________________________________')

    print("i", i)

    data = EvalData(25, 80)
    i = 0
    while (1):
        i += 1
        x, x_len, a, y = next(data)
        print(len(x), a)

        print('______________________________________________________________________')
        break

    print("i", i)
