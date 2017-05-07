import tensorflow as tf
import numpy as np

from model import AspectLevelModel

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.set_random_seed(1)
    resume_from_checkpoint = False
    with tf.Session() as session:
        hidden_size = 128
        model = AspectLevelModel('lstm', hidden_size=128, vocab_size=1000, aspect_vocab_size=100, embedding_size=300, debug=True)

        saver = tf.train.Saver()

        if resume_from_checkpoint:
            saver = tf.train.import_meta_graph('saves/model.ckpt.meta')
            saver.restore(session, tf.train.latest_checkpoint('./saves'))
        else:
            session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
