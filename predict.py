import tensorflow as tf
import numpy as np
from model import FastText as fastText
from lib import load_data_predict, load_final_test_data, create_voabulary, create_voabulary_label
from tensorlayer.prepro import pad_sequences
import os
import codecs


FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("label_size",1999,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("num_sampled",100,"number of noise sampling")
tf.app.flags.DEFINE_string("ckpt_dir","fast_text_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",100,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",15,"embedding size")
tf.app.flags.DEFINE_integer("validate_every", 10, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_string("predict_target_file","fast_text_checkpoint/zhihu_result_ftB2.csv","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'test-zhihu6-title-desc.txt',"source file path for final prediction")


def main(_):
    vocabulary_word2index, vocabulary_index2word = create_voabulary(word2vec_model_path='./zhihu-word2vec-title-desc.bin-100')
    vocab_size = len(vocabulary_word2index)
    vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(voabulary_label_path='./train-zhihu4-only-title-all_1.txt')
    questionid_question_lists=load_final_test_data(FLAGS.predict_source_file)
    test = load_data_predict(vocabulary_word2index, vocabulary_word2index_label, questionid_question_lists)
    test_x=[]
    question_id_list=[]
    for tuple in test:
        question_id, question_string_list=tuple
        question_id_list.append(question_id)
        test_x.append(question_string_list)
    print("start padding....")
    test_x2 = pad_sequences(test_x, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    print("end padding...")
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        fast_text=fastText(FLAGS.label_size, FLAGS.learning_rate, FLAGS.batch_size,FLAGS.num_sampled,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            return
        number_of_training_data=len(test_x2)
        print("number_of_training_data:", number_of_training_data)
        batch_size=1
        index=0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data+1, batch_size)):
            logits=sess.run(fast_text.logits, feed_dict={fast_text.sentence: test_x2[start:end]}) #'shape of logits:', ( 1, 1999)
            predicted_labels=get_label_using_logits(logits[0], vocabulary_index2word_label)
            write_question_id_with_labels(question_id_list[index], predicted_labels, predict_target_file_f)
            index=index+1
        predict_target_file_f.close()


def get_label_using_logits(logits, vocabulary_index2word_label, top_number=5):
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    label_list=[]
    for index in index_list:
        label=vocabulary_index2word_label[index]
        label_list.append(label)
    return label_list


def write_question_id_with_labels(question_id,labels_list,f):
    labels_string=",".join(labels_list)
    f.write(question_id+","+labels_string+"\n")

if __name__ == "__main__":
    tf.app.run()
