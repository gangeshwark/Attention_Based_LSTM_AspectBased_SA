import tensorflow as tf
import numpy as np
import h5py

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

        print len(lines)
        for line in lines:
            i, word = line.strip().split('\t')
            emb.append(get_emb(word, h))
        print len(emb), len(emb[0])
        emb = np.asarray(emb)
        print emb.shape

    with open('../data/aspect_vocab.vocab', 'r') as f:
        h = h5py.File('../data/aspect_vector.hdf5')
        lines = f.readlines()
        print len(lines)
        for line in lines:
            i, word = line.strip().split('\t')
            a_emb.append(get_emb(word, h))
        print len(a_emb), len(a_emb[0])
        a_emb = np.asarray(a_emb)
        print a_emb.shape
    return emb, a_emb, emb.shape[0], a_emb.shape[0]


if __name__ == '__main__':
    embedding, aspect_embedding, vocab_size, aspect_vocab_size = load_emb()
    tf.reset_default_graph()
    tf.set_random_seed(1)
    resume_from_checkpoint = False
    with tf.Session() as session:
        hidden_size = 128
        model = AspectLevelModel('lstm', hidden_size=128, vocab_size=vocab_size, aspect_vocab_size=aspect_vocab_size,
                                 embedding_size=300,
                                 aspect_embedding_size=300,
                                 debug=True)

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
