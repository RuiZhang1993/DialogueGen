import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, AttentionCellWrapper
import utils

class Config():
    def __init__(self):

        self.is_training = False
        self.is_sample = False
        self.is_search = False

        self.save_path = "./models/"
        self.save_file = "dialogue-model"
        self.save_step = 20

        self.load_path = "./models/"

        self.embedding_size = 128
        self.encoder_hidden_units = 128
        self.decoder_hidden_units = 128

        self.batch_size = 5

        self.is_attention = False
        self.attn_length = 128

        self.decode_length = 20

        self.vocab_size = 8000
        self.idx_start = -1
        self.idx_eos = -1
        self.idx_pad = -1
        self.idx_unk = -1

        self.cell_type = GRUCell

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

    def train(self):
        vocab_to_idx, idx_to_vocab = utils.load_vocab_dict_en("./dict")
        data = utils.load_data_en("./data/test_data_en.txt", vocab_to_idx)
        cpairs = [([i for i in data[l]],[t for t in data[l+1]]) for l in range(len(data) - 1)]

        print data
        print cpairs

        def next_feed():
            encoder_inputs_, encoder_inputs_len_ = utils.batch_op([p[0] for p in cpairs], vocab_to_idx['<PAD>'])
            decoder_targets_, _ = utils.batch_op([p[1] for p in cpairs], vocab_to_idx['<PAD>'])

            return {
                self.encoder_inputs: encoder_inputs_,
                self.encoder_inputs_length: encoder_inputs_len_,
                self.decoder_targets: decoder_targets_
            }

        loss_track = []

        for b in range(100):
            fd = next_feed()
            _, l = self.sess.run([self.train_op, self.loss], fd)
            loss_track.append(l)

            if b == 0 or b % self.config.save_step == 0:
                print('batch {}'.format(b))
                print(' minibatch loss: {}'.format(loss_track[-1]))
                predict_ = self.sess.run(self.prediction, fd)
                for i, (inp, pred) in enumerate(zip(fd[self.encoder_inputs].T, predict_.T)):
                    print('  sample {}'.format(i+1))
                    print('   target     > {}'.format([idx_to_vocab[w] for w in inp]))
                    print('   prediction > {}'.format([idx_to_vocab[w] for w in pred]))
                    if i >= 2:
                        break

config = Config()
vocab_to_idx, idx_to_vocab = utils.load_vocab_dict_en("./dict")

# ----- Parameters Initialization -----
config.vocab_size = len(vocab_to_idx)
config.idx_start = vocab_to_idx['<START>']
config.idx_eos = vocab_to_idx['<EOS>']
config.idx_pad = vocab_to_idx['<PAD>']
config.idx_unk = vocab_to_idx['<UNK>']

model = Model(config)
model.train()