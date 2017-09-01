import tensorflow as tf
import numpy as np
from model import FastText as fastText
from lib import load_data, create_voabulary, create_voabulary_label
from tensorlayer.prepro import pad_sequences
import random
import word2vec


FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("label_size",1999,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("num_sampled",200,"number of noise sampling")
tf.app.flags.DEFINE_string("ckpt_dir","fast_text_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",128,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",15,'epoch')
tf.app.flags.DEFINE_integer("validate_every", 5, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_string("cache_path","./checkpoint/data_cache.pik","checkpoint location for the model")


def main(_):
    v_word2index, v_index2word = create_voabulary(word2vec_model_path='./zhihu-word2vec-title-desc.bin-100')
    vocab_size = len(v_word2index)
    v_word2index_label, _ = create_voabulary_label(voabulary_label_path='./train-zhihu4-only-title-all_1.txt')
    train, test = load_data(v_word2index, v_word2index_label, training_data_path='./train-zhihu4-only-title-all_1.txt')
    train_x, train_y = train
    test_x, test_y = test
    print(np.array(train_x).shape)
    train_x = pad_sequences(train_x, maxlen=FLAGS.sentence_len, value=0.)
    test_x = pad_sequences(test_x, maxlen=FLAGS.sentence_len, value=0.)

    with tf.Session() as sess:
        fast_text = fastText(FLAGS.label_size, FLAGS.learning_rate, FLAGS.batch_size,
                           FLAGS.num_sampled, FLAGS.sentence_len, vocab_size, FLAGS.embed_size, FLAGS.is_training)
        print('Initializing Variables')
        sess.run(tf.global_variables_initializer())
        curr_epoch=sess.run(fast_text.epoch_step)
        number_of_training_data=len(train_x)
        batch_size=FLAGS.batch_size
        for epoch in range(curr_epoch, FLAGS.num_epochs):  
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                curr_loss, curr_acc, _ = sess.run([fast_text.loss_val, fast_text.accuracy, fast_text.train_op],
                                                  feed_dict={fast_text.sentence:train_x[start:end], fast_text.labels:train_y[start:end]})
                loss, acc, counter=loss+curr_loss, acc+curr_acc, counter+1
                if counter %500==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch, counter, loss/float(counter), acc/float(counter)))
            sess.run(fast_text.epoch_increment)
            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                eval_loss, eval_acc=do_eval(sess, fast_text, test_x, test_y, batch_size)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))
                saver = tf.train.Saver()
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=fast_text.epoch_step)
        #test_loss, test_acc = do_eval(sess, fast_text, test_x, test_y, batch_size)


def do_eval(sess, fast_text, eval_x, eval_y, batch_size):
    number_examples = len(eval_x)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        curr_eval_loss, curr_eval_acc, = sess.run([fast_text.loss_val, fast_text.accuracy],
                                          feed_dict={fast_text.sentence: eval_x[start:end], fast_text.labels: eval_y[start:end]})
        eval_loss, eval_acc, eval_counter=eval_loss+curr_eval_loss, eval_acc+curr_eval_acc, eval_counter+1
    return eval_loss/float(eval_counter), eval_acc/float(eval_counter)

if __name__ == "__main__":
    tf.app.run()
