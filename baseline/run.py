import tensorflow as tf
import numpy as np
import h5py

from data_loader import TrainData, TestData
from prepare_data import get_w2i, get_a2i
from model import AspectLevelModel


def get_emb(word, h):
    if word in h:
        return h[word][:]
    else:
        return h['__UNK__'][:]


def load_emb():
    emb = []
    a_emb = []
    with open('../data/text_vocab.vocab', 'r') as f:
        h = h5py.File('../data/text_vector.hdf5')
        lines = f.readlines()

        for line in lines:
            i, word = line.strip().split('\t')
            emb.append(get_emb(word, h))
        emb = np.asarray(emb)

    with open('../data/aspect_vocab.vocab', 'r') as f:
        h = h5py.File('../data/aspect_vector.hdf5')
        lines = f.readlines()

        for line in lines:
            i, word = line.strip().split('\t')
            a_emb.append(get_emb(word, h))
        #print len(a_emb), len(a_emb[0])
        a_emb = np.asarray(a_emb)
        #print a_emb.shape
    return emb, a_emb, emb.shape[0], a_emb.shape[0]


def convert_ids_sent(ids, i2w):
    sents = []
    sent = []
    for x in ids:
        for i in x:
            if i in i2w:
                sent.append(i2w[x])
            else:
                sent.append('__UNK__')
        sents.append(sent)
    # print id
    return sents


if __name__ == '__main__':
    w2i, i2w = get_w2i()
    a2i, i2a = get_a2i()
    embedding, aspect_embedding, vocab_size, aspect_vocab_size = load_emb()
    tf.reset_default_graph()
    tf.set_random_seed(1)
    resume_from_checkpoint = False
    with tf.Session() as session:
        hidden_size = 256
        batch_size = 32
        #infered from the dataset
        input_len = 80
        model = AspectLevelModel('lstm', hidden_size=hidden_size, vocab_size=vocab_size,
                                 aspect_vocab_size=aspect_vocab_size,
                                 embedding_size=300,
                                 aspect_embedding_size=300,
                                 debug=False, input_length=input_len, batch_size=batch_size)

        saver = tf.train.Saver()

        if resume_from_checkpoint:
            saver = tf.train.import_meta_graph('saves/model.ckpt.meta')
            saver.restore(session, tf.train.latest_checkpoint('./saves'))
        else:
            session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        session.run([model.embedding_init, model.aspect_embedding_init],
                    feed_dict={model.embedding_placeholder: embedding,
                               model.aspect_embedding_placeholder: aspect_embedding})
        loss = []
        train_data = TrainData(batch_size=batch_size, input_len=input_len)
        test_data = TestData(batch_size=batch_size, input_len=input_len)
        try:
            b_i = 0
            while (True):
                x, x_len, a, y = next(train_data)

                if x.shape[0] <= 0:
                    print "Training complete!"
                    break

                print type(x[0][0]), type(a[0]), type(y[0]), type(x_len[0])
                fd = {
                    model.inputs: x,
                    model.inputs_length: x_len,
                    model.input_aspect: a,
                    model.targets: y,
                }
                _, l = session.run([model.train_op, model.loss], feed_dict=fd)
                loss.append(l)

                if b_i % 5:
                    minibatch_loss = session.run([model.loss], fd)
                    x, x_len, a = next(test_data)
                    if x.shape[0] <= 0:
                        print "No more data to test with"
                        continue

                    fd = {
                        model.inputs: np.asarray(x),
                        model.inputs_length: np.asarray(x_len),
                        model.input_aspect: np.asarray(a),
                    }
                    inference = session.run(model.logits_train, fd)
                    input = fd[model.inputs]
                    input_aspect = fd[model.input_aspect]
                    target = model.targets
                    print "Review: ", convert_ids_sent(input, i2w)
                    print "Aspect: ", [i2a[i] for i in input_aspect]
                    print target, inference
                b_i += 1

        except KeyboardInterrupt:
            print "Training Interrupted"

        import matplotlib.pyplot as plt

        plt.plot(loss)
