import tensorflow as tf
import numpy as np
import h5py
from time import time
from tqdm import tqdm

from baseline.data_loader import TrainData, EvalData
from baseline.model import AspectLevelModel
from baseline.prepare_data import get_w2i, get_a2i


def get_emb(word, h):
    if word in h:
        return h[word][:]
    else:
        return h['__UNK__'][:]


def load_emb():
    emb = []
    a_emb = []
    with open('../data/semeval14/text_vocab.vocab', 'r') as f:
        h = h5py.File('../data/semeval14/text_vector.hdf5')
        lines = f.readlines()

        for line in lines:
            i, word = line.strip().split('\t')
            emb.append(get_emb(word, h))
        emb = np.asarray(emb)

    with open('../data/semeval14/aspect_vocab.vocab', 'r') as f:
        h = h5py.File('../data/semeval14/aspect_vector.hdf5')
        lines = f.readlines()

        for line in lines:
            i, word = line.strip().split('\t')
            a_emb.append(get_emb(word, h))
        # print len(a_emb), len(a_emb[0])
        a_emb = np.asarray(a_emb)
        # print a_emb.shape
    return emb, a_emb, emb.shape[0], a_emb.shape[0]


def convert_ids_sent(ids, i2w):
    sent = []
    for x in ids:
        if str(x) in i2w:
            sent.append(i2w[str(x)])
        else:
            sent.append('__UNK__')
    # print id
    return sent


if __name__ == '__main__':
    w2i, i2w = get_w2i()
    print('Len i2w', len(i2w))
    a2i, i2a = get_a2i()
    embedding, aspect_embedding, vocab_size, aspect_vocab_size = load_emb()
    tf.reset_default_graph()
    tf.set_random_seed(1)
    resume_from_checkpoint = False
    with tf.Session() as session:
        hidden_size = 300
        batch_size = 25
        # infered from the dataset
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
        st = time()
        try:
            print("Training")

            for epoch in range(1000):
                # print "Epoch: ", epoch
                train_data = TrainData(batch_size=batch_size, input_len=input_len)
                # test_data = TestData(batch_size=batch_size, input_len=input_len)
                test_data = EvalData(batch_size=batch_size, input_len=input_len)
                tq = tqdm(range(140))
                for batch in tq:

                    x, x_len, a, y = next(train_data)

                    if x.shape[0] < batch_size:
                        # print "Training complete!"
                        break

                    # print type(x[0][0]), type(a[0]), type(y[0]), type(x_len[0])

                    fd = {
                        model.inputs: x,
                        model.inputs_length: x_len,
                        model.input_aspect: a,
                        model.targets: y,
                    }
                    _, l = session.run([model.train_op, model.loss], feed_dict=fd)
                    loss.append(l)

                    if batch % 10 == 0:
                        minibatch_loss = session.run([model.loss], fd)

                        x, x_len, a, y = next(test_data)
                        if x.shape[0] < batch_size:
                            print("No more data to test with")
                            continue


                        def accuracy(predictions, labels):
                            diff = []
                            for y, y_ in zip(labels, predictions):
                                diff.append(1.00 * np.sum(np.argmax(y) == np.argmax(y_)))
                            # print "Diff: ", diff
                            return (sum(diff) / len(diff))


                        fd = {
                            model.inputs: np.asarray(x),
                            model.inputs_length: np.asarray(x_len),
                            model.input_aspect: np.asarray(a),
                        }
                        inference = session.run(model.logits_train, fd)
                        tq.set_description("Epoch:%d,  Minibatch loss: %s, Accuracy: %s" % (
                            epoch+1, minibatch_loss[0], accuracy(inference, y)))
                        input = fd[model.inputs]
                        input_aspect = fd[model.input_aspect]
                        # print "Review: ", x[:2], input.shape
                        c, d = batch_size - 2, batch_size
                        # print "Len: ", x_len[c:d]
                        m = [' '.join(convert_ids_sent(x1, i2w)) for x1 in x[c:d]]
                        # print "Review: ", input.shape  # ,convert_ids_sent(input, i2w)
                        # for n in m:
                        # print "\n", n
                        # print "Aspect: ", [i2a[str(i)] for i in input_aspect[c:d]]
                        # print "Class", [a for a in inference[c:d]]
            print("Training complete!")
            print("Training Time: ", time() - st, " seconds")
        except KeyboardInterrupt:
            print("Training Time: ", time() - st, " seconds")
            print("Training Interrupted")

        import matplotlib.pyplot as plt

        plt.plot(loss)
