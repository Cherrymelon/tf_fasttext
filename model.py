import tensorflow as tf
import numpy as np


class FastText:
    def __init__(self, label_size, learning_rate, batch_size, num_sampled, sentence_len, vocab_size, embed_size, is_training):
        """init all hyperparameter here"""
        self.label_size = label_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.sentence_len = sentence_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate


        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_len], name="sentence")  # X
        self.labels = tf.placeholder(tf.int32, [None], name="Labels")  # y
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.instantiate_weights()
        self.logits = self.inference()  # [None, self.label_size]
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

    def instantiate_weights(self):
        self.Embedding = tf.get_variable("Embedding", [self.vocab_size, self.embed_size])
        self.W = tf.get_variable("W", [self.embed_size, self.label_size])
        self.b = tf.get_variable("b", [self.label_size])

    def inference(self):
        sentence_embeddings = tf.nn.embedding_lookup(self.Embedding, self.sentence)  # [None,self.sentence_len,self.embed_size]
        self.sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1)  # [None,self.embed_size]
        logits = tf.matmul(self.sentence_embeddings, self.W) + self.b #[None, self.label_size]==tf.matmul([None,self.embed_size],[self.embed_size,self.label_size])
        return logits

    def loss(self):
        if self.is_training: #training
            labels = tf.reshape(self.labels, [-1])
            labels = tf.expand_dims(labels, 1)
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=tf.transpose(self.W),
                               biases=self.b,
                               labels=labels,
                               inputs=self.sentence_embeddings,
                               num_sampled=self.num_sampled,
                               num_classes=self.label_size,
                               partition_strategy="div"))
        else:
            labels_one_hot = tf.one_hot(self.labels, self.label_size) #[batch_size]---->[batch_size,label_size]
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=self.logits) #labels:[batch_size,label_size];logits:[batch, label_size]
            print("loss0:", loss) #shape=(?, 1999)
            loss = tf.reduce_sum(loss, axis=1)
            print("loss1:", loss)  #shape=(?,)
        return loss

    def train(self):
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=self.learning_rate, optimizer="Adam")
        return train_op


def tt():
    num_classes=19
    learning_rate=0.01
    batch_size=8
    decay_steps=1000
    decay_rate=0.9
    sequence_length=10
    vocab_size = 10000
    embed_size = 100
    is_training=True
    dropout_keep_prob=1
    fastext=FastText(num_classes, learning_rate, batch_size, decay_steps, decay_rate, 5, sequence_length, vocab_size, embed_size, is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.zeros((batch_size, sequence_length), dtype=np.int32) #[None, self.sequence_length]
            input_y = np.array([1, 0, 1, 1, 1, 2, 1, 1], dtype=np.int32) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
            loss, acc, predict, _ = sess.run([fastext.loss_val, fastext.accuracy, fastext.predictions, fastext.train_op],
                                        feed_dict={fastext.sentence:input_x, fastext.labels:input_y})
            print("loss:", loss, "acc:", acc, "label:", input_y, "prediction:", predict)

#print("ended...")
