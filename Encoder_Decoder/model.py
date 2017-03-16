# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, AttentionCellWrapper
import utils
from math import ceil
from numpy import random

class Config():
    def __init__(self, dictionary_path="./dict/dict_en.voc"):

        self.is_training = False
        self.is_sample = False
        self.is_search = False

        self.is_save = False
        self.save_file = "./models/dialogue-model"
        self.save_step = 20

        self.load_path = "./models/dialogue-model"

        self.corpus_path = "./data/test_data_en.txt"

        self.embedding_size = 128
        self.encoder_hidden_units = 128
        self.decoder_hidden_units = 128

        self.batch_size = 2
        self.batch_epoch = 100

        self.is_attention = False
        self.attn_length = 128

        self.decode_length = 20

        self.cell_type = GRUCell

        if "_en" in dictionary_path:
            self.vocab_to_index, self.index_to_vocab = utils.load_vocab_dict_en(dictionary_path)
            self.vocab_size = len(self.vocab_to_index)
            self.idx_start = self.vocab_to_index['<START>']
            self.idx_eos = self.vocab_to_index['<EOS>']
            self.idx_pad = self.vocab_to_index['<PAD>']
            self.idx_unk = self.vocab_to_index['<UNK>']
        else:
            raise NotImplementedError('Multi-languages support not implemented yet.')


class Model(object):

    def __init__(self, config):
        self.config = config
        self.sess = tf.Session()
        self._construct()


    def _construct(self):

        config_ = self.config
        cell_ = self.config.cell_type

        self.encoder_inputs = tf.placeholder(dtype=tf.int32,
                                             shape=(None, None),
                                             name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32,
                                                    shape=(None,),
                                                    name='encoder_inputs_length')
        self.decoder_targets = tf.placeholder(dtype=tf.int32,
                                              shape=(None, None),
                                              name='decoder_targets')

        with tf.device("/cpu:0"):
            self.embeddings = tf.Variable(
                tf.random_uniform([self.config.vocab_size, config_.embedding_size],
                                  -1.0, 1.0),
                dtype=tf.float32)

        self.start_slice = tf.ones([config_.batch_size], dtype=tf.int32, name='START') * config_.idx_start
        self.start_embed = tf.nn.embedding_lookup(self.embeddings, self.start_slice)

        self.pad_slice = tf.ones([config_.batch_size], dtype=tf.int32, name='PAD') * config_.idx_pad
        self.pad_embed = tf.nn.embedding_lookup(self.embeddings, self.pad_slice)

        self.eos_slice = tf.ones([config_.batch_size], dtype=tf.int32, name='EOS') * config_.idx_eos
        self.eos_embed = tf.nn.embedding_lookup(self.embeddings, self.eos_slice)

        with tf.variable_scope("encoder"):

            self.encoder_inputs_embed = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

            self.encoder_cell = cell_(num_units=config_.encoder_hidden_units)

            if config_.is_attention:
                self.encoder_cell = AttentionCellWrapper(self.encoder_cell,
                                                         attn_length=config_.attn_length,
                                                         state_is_tuple=True)

            self.encoder_outputs,self.encoder_final_state = tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                                                              inputs=self.encoder_inputs_embed,
                                                                              dtype=tf.float32,
                                                                              time_major=True)

        with tf.variable_scope('decoder'):

            self.decoder_cell = cell_(num_units=config_.decoder_hidden_units)

            if config_.is_attention:
                self.decoder_cell = AttentionCellWrapper(self.decoder_cell,
                                                         attn_length=config_.attn_length,
                                                         state_is_tuple=True)
            self.W_decoder = tf.Variable(
                tf.random_uniform([config_.decoder_hidden_units, config_.vocab_size],
                                  -1.0, 1.0),
                dtype=tf.float32)

            self.b_decoder = tf.Variable(tf.zeros([config_.vocab_size], dtype=tf.float32))

            self.decoder_length, _ = tf.unstack(tf.shape(self.decoder_targets))
            self.decoder_length = self.decoder_length + 0

            def decoder_loop_fn_init():
                initial_element_finished = (0 >= self.decoder_length)
                initial_input = self.start_embed
                initial_cell_state = self.encoder_final_state
                initial_cell_output = None
                initial_loop_state = None
                return (initial_element_finished,
                        initial_input,
                        initial_cell_state,
                        initial_cell_output,
                        initial_loop_state)

            def decoder_loop_fn_trans(time, previous_output, previous_state, previous_loop_state):

                def get_next_input():
                    output_logits = tf.add(tf.matmul(previous_output, self.W_decoder), self.b_decoder)
                    prediction = tf.argmax(output_logits, axis=1)
                    next_input = tf.nn.embedding_lookup(self.embeddings, prediction)
                    return next_input

                elements_finished = (time >= self.decoder_length)
                finished = tf.reduce_all(elements_finished)
                input = tf.cond(finished, lambda: self.pad_embed, get_next_input)
                state = previous_state
                output = previous_output
                loop_state = None

                return (elements_finished, input, state, output, loop_state)

            def decoder_loop_fn(time, previous_output, previous_state, previous_loop_state):
                if previous_state is None:
                    assert previous_output is None
                    return decoder_loop_fn_init()
                else:
                    return decoder_loop_fn_trans(time,
                                                 previous_output,
                                                 previous_state,
                                                 previous_loop_state)

            decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(self.decoder_cell, decoder_loop_fn)

            self.decoder_outputs = decoder_outputs_ta.stack()
            decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(self.decoder_outputs))

            decoder_outputs_flat = tf.reshape(self.decoder_outputs, (-1, decoder_dim))
            decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, self.W_decoder), self.b_decoder)
            self.decoder_logits = tf.reshape(decoder_logits_flat,
                (decoder_max_steps, decoder_batch_size, config_.vocab_size))

        self.prediction = tf.argmax(self.decoder_logits, 2)

        self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=config_.vocab_size, dtype=tf.float32),
            logits=self.decoder_logits)

        self.loss = tf.reduce_mean(self.stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def train(self):
        self.data, batches = self.__load_data()
        loss_track = []
        for batch in range(self.config.batch_epoch):
            random.shuffle(self.data)
            for index in range(batches):
                fd = self.__next_feed(index)
                _, l = self.sess.run([self.train_op, self.loss], fd)
                loss_track.append(l)

            if batch == 0 or batch % self.config.save_step == 0:
                self.__print_comparation(fd, batch)
                if self.config.is_save:
                    self.__save(batch)

    def __load_data(self):
        data = utils.load_data_en(self.config.corpus_path, self.config.vocab_to_index)
        data = [([i for i in data[l]],[t for t in data[l+1]]) for l in range(len(data) - 1)]
        batches = int(ceil(float(len(data) // self.config.batch_size)))
        return data, batches

    def __next_feed(self, index):
        idx_pad = self.config.vocab_to_index['<PAD>']

        start_ = index * self.config.batch_size
        end_ = (index+1) * self.config.batch_size

        encoder_inputs_, encoder_inputs_len_ = utils.batch_op([p[0] for p in self.data[start_:end_]], idx_pad)
        decoder_targets_, _ = utils.batch_op([p[1] for p in self.data[start_:end_]], idx_pad)
        return {
            self.encoder_inputs: encoder_inputs_,
            self.encoder_inputs_length: encoder_inputs_len_,
            self.decoder_targets: decoder_targets_
        }

    def __print_comparation(self, feed_dict, batch):
        idx2voc = self.config.index_to_vocab
        print('batch {}'.format(batch))
        print(' minibatch loss: {}'.format(self.sess.run(self.loss, feed_dict)))
        predict_ = self.sess.run(self.prediction, feed_dict)
        for i, (inp, pred) in enumerate(zip(feed_dict[self.encoder_inputs].T, predict_.T)):
            print('  sample {}'.format(i + 1))
            print('   target     > {}'.format([idx2voc[w] for w in inp]))
            print('   prediction > {}'.format([idx2voc[w] for w in pred]))
            if i >= 2:
                break

    def __save(self, id):
        self.saver.save(self.sess, self.config.save_file, global_step=id)

    def __load(self, meta_graph):
        new_saver = tf.train.import_meta_graph(meta_graph)
        new_saver.restore(self.sess, tf.train.latest_checkpoint(self.config.load_path))

config = Config()

model = Model(config)
model.train()
